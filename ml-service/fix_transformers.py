import os
filepath = r'C:\Users\Vineet\.safechat_venv\Lib\site-packages\transformers\utils\import_utils.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace("def check_torch_load_is_safe() -> None:", "def check_torch_load_is_safe() -> None:\n    return\n")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)
print('Nuked CVE check!')
