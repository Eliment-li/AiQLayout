import re
import numpy as np

def parse_phrase(instruction):
    # 使用正则表达式匹配括号内的内容
    match = re.search(r'\((.*?)\)', instruction)
    if not match:
        raise ValueError("输入字符串中没有找到括号")

    expr = match.group(1).replace(' ', '')  # 移除空格

    # 处理 pi 的表达式
    pi_pattern = re.compile(r'^([+-]?[\d\.]*)\*?pi(?:/([+-]?[\d\.]+))?$')
    m = pi_pattern.match(expr)
    if m:
        m_coeff = m.group(1)
        n_denom = m.group(2)
        # 处理 m
        if m_coeff == '' or m_coeff == '+':
            m_val = 1.0
        elif m_coeff == '-':
            m_val = -1.0
        else:
            m_val = float(m_coeff)
        # 处理 n
        if n_denom:
            n_val = float(n_denom)
            return m_val * np.pi / n_val
        else:
            return m_val * np.pi

    # 纯数字
    try:
        return float(expr)
    except ValueError:
        raise ValueError(f"invalid expr: {expr}")


if __name__ == '__main__':
    # 测试用例
    for test in [
        "p(pi)", "p(-pi)", "p(pi/3)", "p(-pi/7)", "p(2*pi/5)", "p(-2*pi/5)", "p(3*pi)", "p(4)", "p(-1.5)", "p(0)",
        "p(+pi/2)", "p(-pi/2)", "p(+3*pi/4)", "p(3.5*pi/2)", "p(2.1)"
    ]:
        print(f"{test} -> {parse_phrase(test)}")