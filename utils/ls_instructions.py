import numpy as np
import os
def print_ins(instructions):
    for ins in instructions:
        if len(ins)>0:
            print(ins)

import re

def convert_plus_and_measure(instructions,i):
    line = instructions[i]
    if  i + 2 < len(instructions):
        next_line1 = instructions[i + 1].strip()
        next_line2 = instructions[i + 2].strip()
        if next_line1.startswith('MultiBodyMeasure') and next_line2.startswith('MultiBodyMeasure'):
            # 提取Init的数字
            match1 = re.match(r'Init (\d+) \|(\+|0)>', line)
            # 提取MultiBodyMeasure的参数
            match2 = re.match(r'MultiBodyMeasure (\d+):Z,(\d+):[Z,X]', next_line1)
            match3 = re.match(r'MultiBodyMeasure (\d+):[Z,X],(\d+):[Z,X]', next_line2)
            if match1 and match2 and match3:
                patch_id = match1.group(1)
                patch_id_1 = match2.group(1)
                patch_id_2 = match2.group(2)
                patch_id_3 = match3.group(1)
                patch_id_4 = match3.group(2)
                assert patch_id == patch_id_2 and patch_id==patch_id_4, "Mismatched patch IDs in Init and MultiBodyMeasure"
                # 返回转换后的行
                line =  f"Plus_Mear_Two {patch_id_1}:Z,{patch_id_3}:X"
                return line, i + 3

    inst = None
    return inst, i + 1  # 如果没有匹配，返回 None 和下一个索引


def process_file(input_file, output_file):
    # 定义需要删除的行开头
    prefixes_to_remove = ['SGate', 'HGate', 'LogicalPauli', 'MeasureSinglePatch']
    instructions = []
    with open(input_file, 'r') as infile:
        qubits_number =15 #re.search(r"_(\d+)\.txt$", input_file).group(1)
        lines = infile.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 检查是否需要删除该行
            should_remove = any(line.startswith(prefix) for prefix in prefixes_to_remove)
            if should_remove:
                i += 1
                continue
            # 检查是否是RequestMagicState后跟MultiBodyMeasure的情况
            if line.startswith('RequestMagicState') and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('MultiBodyMeasure'):
                    # 提取RequestMagicState的数字
                    match1 = re.match(r'RequestMagicState (\d+)', line)
                    # 提取MultiBodyMeasure的参数
                    match2 = re.match(r'MultiBodyMeasure (\d+):Z,(\d+):X', next_line)
                    if match1 and match2:
                        m_state_number = match1.group(1)
                        patch_id = match2.group(1)
                        m_state_number_2 = match2.group(2)
                        assert m_state_number == m_state_number_2, "Mismatched patch IDs in RequestMagicState and MultiBodyMeasure"
                        # 写入转换后的行
                        #outfile.write(f"Request_M {patch_id}:Z,M:X\n")
                        # print(f"Request_M {patch_id}:Z,M:X\n")
                        instructions.append(f"Request_M {patch_id}:Z,M:X")
                        i += 2
                    else:
                        instructions.append(line)
                        i+=1
                else:
                    i += 1
                    print('magic state unused')
            elif line.startswith('Init'):
                inst,i = convert_plus_and_measure(lines,i)
                if inst is  None:
                    print(lines)
                instructions.append(inst)
            else:
                # 如果不是需要特殊处理的行，直接写入
                #outfile.write(line + '\n')
                instructions.append(line)
                i += 1
        #print_ins(instructions)
        ins_cnt = handle_repeated_lines(instructions)
        ins_resign = re_sign_patch_id(ins_cnt)
        # print_ins(ins_resign)
        ins_heat = inst_to_heatmap(ins_resign,int(qubits_number))
        print(repr(ins_heat))
        return ins_heat


'''
group the adjcent repeated lines 
'''
def handle_repeated_lines(arr):
    if not arr:  # 处理空数组情况
        return []
    result = []
    current_element = arr[0]
    count = 1

    for element in arr[1:]:
        if element == current_element:
            count += 1
        else:
            result.append([current_element, count])
            current_element = element
            count = 1

    # 添加最后一个元素的统计
    result.append([current_element, count])
    return result


import re

'''
resign the patch id of patchs
'''
def replace_numbers_in_string(s, replacement_map):
    '''
    ['Init 3706 |+>', 1]
    ['MultiBodyMeasure 8:Z,3706:Z', 1]
    ['MultiBodyMeasure 6:X,3706:X', 1]
    这类命令视为
    Plus_Mear_Two 8:Z,6:X
    '''
    # 使用正则表达式找到所有连续的数字
    pattern = re.compile(r'\d+')

    # 定义一个替换函数，用于处理每个匹配到的数字
    def replace_match(match):
        num_str = match.group()
        # 如果数字在映射中，则替换，否则保持原样
        return str(replacement_map.get(num_str, num_str))

    # 使用sub方法进行替换
    return pattern.sub(replace_match, s)



def re_sign_patch_id(instructions):
    """
    将指令中的patch_id重新签名为从0开始的连续整数
    """
    patch_id_map = {}
    new_instructions = []
    current_patch_id = 1

    for row in instructions:
        inst = row[0]
        if inst is None:
            print('err, inst is none')
            continue
        match1= re.match(r'Request_M (\d+):Z,M:X', inst)
        match2= re.match(r'Init (\d+)*', inst)
        match3 = re.match(r'MultiBodyMeasure (\d+):[Z,X],(\d+):[Z,X]', inst)
        if match1:
            patch_id = match1.group(1)
            if patch_id not in patch_id_map:
                patch_id_map[patch_id] = current_patch_id
                current_patch_id += 1
        elif match2:
            patch_id = match2.group(1)
            if patch_id not in patch_id_map:
                patch_id_map[patch_id] = current_patch_id
                current_patch_id += 1
        elif match3:
            patch_id1 = match3.group(1)
            patch_id2 = match3.group(2)
            if patch_id1 not in patch_id_map:
                patch_id_map[patch_id1] = current_patch_id
                current_patch_id += 1
            if patch_id2 not in patch_id_map:
                patch_id_map[patch_id2] = current_patch_id
                current_patch_id += 1
    # print(patch_id_map)
    for row in instructions:
        inst = row[0]
        cnt = row[1]
        # 替换patch_id为新的连续整数
        new_inst = replace_numbers_in_string(inst, patch_id_map)
        # 将新的指令和计数添加到新列表中
        new_instructions.append([new_inst, cnt])
    return new_instructions


def inst_to_heatmap(instructions,qubits_number):

    heat_map = np.zeros(shape=(qubits_number+1,qubits_number+1)).astype(int)
    for i in range(len(instructions)):
        ins = instructions[i][0]
        cnt = instructions[i][1]
        if ins.startswith('Request_M'):
            match = re.match(r'Request_M (\d+):Z,M:X', ins)
            q1 = int(match.group(1))
            q2 = 0
        elif ins.startswith('Plus_Mear_Two'):
            #TODO count the Z and X
            match = re.match(r'Plus_Mear_Two (\d+):Z,(\d+):X', ins)
            q1 = int(match.group(1))
            q2 = int(match.group(2))
        if q1 >=q2:
            heat_map[q1][q2] +=cnt
        else:
            heat_map[q2][q1] +=cnt
    #drop the first row and last column
    heat_map = heat_map[1:, :-1]
    return heat_map

def get_heat_map(file_path):
    heat_map = process_file(file_path, '')
    return heat_map

''' this fuction:
     1. takes a list of LSInstructions ,each row for one instruction
     2. remove single patch(logical qubit) operations, like measure, init, etc.
     3. convert the following instructions 
         RequestMagicState patchId
         MultiBodyMeasure 0:Z,2:X
         MeasureSinglePatch 2 Z
         to
         [patchId, Request M]
         [MultiBodyMeasure patchId:Z,patchId:X]

     4.
     convert
        'Init 3706 |+>'
        'MultiBodyMeasure 8:Z,3706:Z'
        'MultiBodyMeasure 6:X,3706:X'
     to
        |+> can be put in a patch,which the patch is  in a path that connecting the  8 and 6
        Plus_Mear_Two 8:Z,6:X
     '''
def phrase_ls_instructions(instructions:str):
    pass

if __name__ == '__main__':
    input_directory = r'D:\sync\mqtbench\ls_inst'
    output_directory = 'D:\sync\mqtbench\out'
    for filename in os.listdir(input_directory):
        print(f'================== process file {filename} ===================================')
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename)
        process_file(input_file,output_file)