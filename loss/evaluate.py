from misc import utils

all_class_name = [
    "approach",
    "chase",
    "circle",
    "eat",
    "clean",
    "sniff",
    "up",
    "walk_away",
    "other"]


def OMBD_evaluate(results, epoch, command, log_file):
    map, aps, cap, caps = utils.frame_level_map_n_cap(results)
    out = '[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, command, map)
    out2 = '[Epoch-{}] [IDU-{}] cap: {:.4f}\n'.format(epoch, command, cap)
    print(out)
    print(out2)
    
    if log_file != '':
        with open(log_file, 'a+') as f:
            f.writelines(out)
        for i, ap in enumerate(aps):
            cls_name = all_class_name[i]
            out = '{}: {:.4f}\n'.format(cls_name, ap)
            # print(out)
            with open(log_file, 'a+') as f:
                f.writelines(out)
        with open(log_file, 'a+') as f:
            f.writelines(out2)
        for i, acap in enumerate(caps):
            cls_name = all_class_name[i]
            out2 = '{}: {:.4f}\n'.format(cls_name, acap)
            # print(out)
            with open(log_file, 'a+') as f:
                f.writelines(out2)


if __name__ == '__main__':
    pass
