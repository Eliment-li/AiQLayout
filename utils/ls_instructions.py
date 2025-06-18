
def ls_instructions_to_dependency():
    pass
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
     '''


import re


def process_file(input_file, output_file):
    # 定义需要删除的行开头
    prefixes_to_remove = ['SGate', 'HGate', 'LogicalPauli', 'MeasureSinglePatch','Init']

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        for i in range(len(lines)):
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
                        print(f"Request_M {patch_id}:Z,M:X\n")
                        i += 2  # 跳过下一行，因为已经处理了

            else:
                # 如果不是需要特殊处理的行，直接写入
                #outfile.write(line + '\n')
                print(line + '\n')
                i += 1


# 使用示例
input_filename = 'd:/ls.txt'  # 替换为你的输入文件名
output_filename = 'd:/out.txt'  # 替换为你的输出文件名
process_file(input_filename, output_filename)
