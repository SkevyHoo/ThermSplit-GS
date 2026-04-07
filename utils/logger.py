
import tqdm
import torch
class Logger:
    
    def __init__(self, progress_bar:tqdm, ema_weight:float=0.6):
        self.pbar = progress_bar
        self.ema_weight = ema_weight

        self.log_dic = {}

    def update(self, display_dict:dict):
        
        for key, (newval,logtype,fmt) in display_dict.items():
            if torch.is_tensor(newval):
                newval = newval.item()
            if logtype.strip().lower() == "ema":    #  EMA计算: 新值 = 旧值 * ema_weight + 新值 * (1-ema_weight)
                self.log_dic[key] = (self.log_dic.get(key,(0,fmt))[0] * self.ema_weight + newval * (1.0-self.ema_weight),fmt)
            elif logtype.strip().lower() == "update": # 直接更新为新值
                self.log_dic[key] = (newval,fmt)
            else:
                raise NotImplementedError

    def show(self): # 格式化所有日志条目并设置到进度条的后缀中
        self.pbar.set_postfix({k:format(val,fmt) for k,(val,fmt) in self.log_dic.items()})
        self.pbar.update(10)