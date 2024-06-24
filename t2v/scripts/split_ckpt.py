import torch

def split_qkv(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'qkv' in key:
            prefix, suffix = key.split('.qkv.')
            q_key = prefix + '.q.' + suffix
            k_key = prefix + '.k.' + suffix
            v_key = prefix + '.v.' + suffix
            print(q_key,k_key,v_key)
            new_state_dict[q_key] = value[:value.size(0) // 3]
            new_state_dict[k_key] = value[value.size(0) // 3: 2 * (value.size(0) // 3)]
            new_state_dict[v_key] = value[2 * (value.size(0) // 3):]
        else:
            new_state_dict[key] = value
    return new_state_dict

state_dict = torch.load('./logs/split_ckpt/OpenSora-v1-HQ-16x512x512.pth')  # your path of the original ckpt

    
new_state_dict = split_qkv(state_dict)


torch.save(new_state_dict, './logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split-test.pth')  # split the qkv layer in the ckpt
