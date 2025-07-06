def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path,load_model_name):
    if load_model_name == '': # original poindp
        print(f'Recovering model and checkpoint from {model_path}')
        checkpoint = torch.load(model_path)

        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model = nn.DataParallel(model)
                model.load_state_dict(checkpoint['model_state'])
                model = model.module

        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # for backward compatibility with saved models
        if 'lr_sched_state' in checkpoint:
            lr_sched.load_state_dict(checkpoint['lr_sched_state'])
            if checkpoint['bnm_sched_state'] is not None:
                bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
        else:
            print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

        return model
    elif load_model_name == 'PCT':
        print('Recovering model and checkpoint from {PCT_Pytorch/checkpoints/best/models/model.t7},'
              'ae is pretrained & commly used')

        if model.is_ae == True:
            ##### PCT clean
            pct_clean_pretrained = torch.load('PCT_Pytorch/checkpoints/best/models/model.t7')
            net_dict = {}
            for k, v in pct_clean_pretrained.items():
                k = k.replace("module.", "")
                net_dict[k] = v
            model.model.load_state_dict(net_dict)

            #####
            checkpoint2 = torch.load('runs/dgcnn_cls_ae_diffusion_1/model_best_test.pth')
            net_dict2 = checkpoint2['model_state']
            prefix_to_remove = 'model.'
            state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
            prefix_to_remove = 'ae.'
            state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
            model.ae.load_state_dict(state_dict) # 加载 pretrained ae

        else:
            ##### PCT clean
            pct_clean_pretrained = torch.load('PCT_Pytorch/checkpoints/best/models/model.t7')
            net_dict = {}
            for k, v in pct_clean_pretrained.items():
                k = k.replace("module.", "model.")
                net_dict[k] = v
            model.load_state_dict(net_dict)

        return model
    elif load_model_name == 'Curvenet':
        print('Recovering model and checkpoint from {pretrained/curvenet/model.t7},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/curvenet/model.t7')

        ### curvenet
        curvenet_clean_pretrained = checkpoint
        net_dict = {}
        for k, v in curvenet_clean_pretrained.items():
            k = k.replace("module.", "")
            net_dict[k] = v
        model.model.load_state_dict(net_dict)

        checkpoint2 = torch.load('runs/dgcnn_cls_ae_diffusion_1/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)

        return model
    elif load_model_name == 'dgcnn':
        print(f'Recovering model and checkpoint from {model_path}')
        checkpoint = torch.load(model_path)

        dgcnn_clean_pretrained = torch.load('runs/model.1024.t7')
        net_dict = {}
        for k, v in dgcnn_clean_pretrained.items():
            k = k.replace("module.", "model.")
            net_dict[k] = v

        for key in net_dict:
            if key in checkpoint['model_state']:
                checkpoint['model_state'][key] = net_dict[key]

        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state'])
            else:
                model = nn.DataParallel(model)
                model.load_state_dict(checkpoint['model_state'])  # ,strict=False)
                model = model.module

        optimizer.load_state_dict(checkpoint['optimizer_state'])

        # for backward compatibility with saved models
        if 'lr_sched_state' in checkpoint:
            lr_sched.load_state_dict(checkpoint['lr_sched_state'])
            if checkpoint['bnm_sched_state'] is not None:
                bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
        else:
            print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

        return model
    elif load_model_name == 'pointmlp':
        print('Recovering model and checkpoint from { },'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/pointMLP/best_checkpoint.pth')

        ### pointMLP
        net_dict = checkpoint['net']
        prefix_to_remove = 'module.'
        net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
        model.model.load_state_dict(net_dict)

        checkpoint2 = torch.load('runs/dgcnn_cls_ae_diffusion_step_1/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)

        return model
    elif load_model_name == 'pointnet':
        print('Recovering model and checkpoint from {pretrained/dgcnn_pointnet_run_1/model_best_test.pth},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/dgcnn_pointnet_run_1/model_best_test.pth')

        if model.is_ae == True:
            ### pointnet
            net_dict = checkpoint['model_state']
            prefix_to_remove = 'module.model.'
            net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
            model.model.load_state_dict(net_dict)  # clean_pretrained pointnet

            checkpoint2 = torch.load('runs/dgcnn_cls_ae_diffusion_1/model_best_test.pth')
            net_dict2 = checkpoint2['model_state']
            prefix_to_remove = 'model.'
            state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
            prefix_to_remove = 'ae.'
            state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
            model.ae.load_state_dict(state_dict)  # pretrained ae
        else:
            net_dict = checkpoint['model_state']
            prefix_to_remove = 'module.model.'
            net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
            model.model.load_state_dict(net_dict)  # clean_pretrained pointnet

        return model
    elif load_model_name == 'pointnet2':
        print('Recovering model and checkpoint from {pretrained/dgcnn_pointnet2_run_1/model_best_test.pth},'
              'ae is pretrained & commly used')
        checkpoint = torch.load('pretrained/dgcnn_pointnet2_run_1/model_best_test.pth')

        ### pointnet
        net_dict = checkpoint['model_state']
        prefix_to_remove = 'module.model.'
        net_dict = {key[len(prefix_to_remove):]: value for key, value in net_dict.items()}
        model.model.load_state_dict(net_dict)  # clean_pretrained pointnet2

        checkpoint2 = torch.load('runs/dgcnn_cls_ae_diffusion_1/model_best_test.pth')
        net_dict2 = checkpoint2['model_state']
        prefix_to_remove = 'model.'
        state_dict = {k: v for k, v in net_dict2.items() if not k.startswith(prefix_to_remove)}
        prefix_to_remove = 'ae.'
        state_dict = {key[len(prefix_to_remove):]: value for key, value in state_dict.items()}
        model.ae.load_state_dict(state_dict)  # pretrained ae

        return model
