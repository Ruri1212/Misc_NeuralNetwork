import os
import time
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, models, transforms

from PIL import Image
import matplotlib.pyplot as plt


import logging
import hydra
from omegaconf import OmegaConf, open_dict,DictConfig
import mlflow
from mlflow.models import infer_signature


def visualize_model(device,output_dir,model,dataloaders,class_names,num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f"Pd: {class_names[preds[j]]} // GT: {class_names[labels[j]]}")

                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                ax.imshow(inp)

                if images_so_far == num_images:
                    plt.savefig(output_dir + '/output.png')  # 画像をファイルに保存
                    break
            break     
    model.train(mode=was_training)     




@hydra.main(config_path="conf",config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print("starting")

    ############### データ処理 ################

    # 訓練データ用のデータ拡張と正規化
    # 検証データ用には正規化のみ実施
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'

    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.epoch


    print("loading data.............")

    ### 画像データの読み込み (ラベルは数字)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                    for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
                
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes




    ############### モデルの構築 ################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = models.resnet18(weights = "IMAGENET1K_V1")
    model = models.resnet18()    
    # 各出力サンプルのサイズは2に設定
    num_ftrs = model.fc.in_features    
    model.fc = nn.Linear(num_ftrs, len(class_names))


    ## モデルをloadする場合
    state_dict_uri = "file:///mnt/vmlqnap02/home/inoue/deeplearning/image_process/mlruns/878525502675615922/133c8ebecf36481984797bab2609b5db/artifacts/model_state_dict"
    state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
    model.load_state_dict(state_dict)
    model = model.to(device)


    # パラメータの更新を行うかどうかを設定
    # for param in model.parameters():        
    #     param.requires_grad = True



    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.learning_rate, momentum=cfg.optimizer.momentum)
    # 7エポックごとに学習率を1/10ずつ減衰
    optim_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    


    ############### ログの設定 ################

    # Hydraの実行ファイルのパスを取得
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # logファイルの作成 (main.log が作成される)
    log = logging.getLogger(__name__)
    # Fileハンドラクラスをインスタンス化
    fl_handler = logging.FileHandler(filename=output_dir + "/exec.log", encoding="utf-8")
    fl_handler.setLevel(logging.DEBUG)
    # インスタンス化したハンドラをそれぞれログ太郎に渡す
    log.addHandler(fl_handler)
    log.info(f"output_dir: {output_dir} \n")
   

    # ネットワークの構造を表示
    # パラメータの数、パラメータの名前、パラメータの勾配の有無を表示
    # for params,name in zip(model.parameters(),model.state_dict().keys()):
    #     log.info(f"{name.ljust(45)} {str(params.shape).replace('torch.Size','').ljust(20)} {str(params.requires_grad).ljust(5)}")


    # mlflowの設定
    with open_dict(cfg):
        cfg.mlflow["tracking_uri"] = "file://" + hydra.utils.get_original_cwd() + "/mlruns"
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = cfg.mlflow.experiment_name
    # experiment_id = "image_process_bee_vs_ant"

    mlflow.set_experiment(experiment_name=experiment_name)

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    log.info(f"experiment_id: {experiment_id}")


    ############### 学習処理 ################    
    with mlflow.start_run(run_name=cfg.mlflow.run_name):        

        best_acc = 0.0
        # best_model_wts = copy.deepcopy(model.state_dict()) 


        # パラメータを保存(辞書式を分解して保持)
        for k in cfg.keys():
            if k != "mlflow":
                for v in cfg[k].items():
                    mlflow.log_param(v[0],v[1])

        mlflow.set_tag("release.version", "abc")        
       


        for epoch in range(num_epochs):                                
                    
            train_acc, train_loss = 0, 0
            val_acc, val_loss = 0, 0
            n_train, n_test = 0, 0

            since = time.time()
                       
            print('Epoch {}/{}'.format(epoch+1,  num_epochs))
            print('-' * 60)

            # 学習フェース
            model.train()
            for inputs,labels in tqdm(dataloaders["train"],desc="train"):
                n_train += len(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)            
                loss.backward()                
                optimizer.step()

                _, predicted = torch.max(outputs, 1)

                # 損失と精度の計算
                train_loss += loss.item()
                train_acc += (predicted == labels).sum()

            optim_scheduler.step()
        
            # 検証フェース
            model.eval()
            with torch.no_grad():
                for inputs,labels in tqdm(dataloaders["val"],desc="val"):
                    n_test += len(labels)

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs,labels)

                    _, predicted = torch.max(outputs, 1)

                    # 損失と精度の計算
                    val_loss += loss.item()
                    val_acc += (predicted == labels).sum()

            #　評価指標の計算と保存
            train_acc = train_acc / n_train
            val_acc = val_acc / n_test
            train_loss = train_loss * batch_size / n_train
            val_loss = val_loss * batch_size / n_test

            mlflow.log_metric("train_loss", train_loss,step=epoch)
            mlflow.log_metric("train_acc", train_acc,step=epoch)
            mlflow.log_metric("val_loss", val_loss,step=epoch)
            mlflow.log_metric("val_acc", val_acc,step=epoch)

            print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
            print("\n\n")

            if val_acc > best_acc:
                best_acc = val_acc
                ## 新しい別のオブジェクトを作成する (代入だと一方の変更が影響を与える)
                best_model_state_dict  = copy.deepcopy(model.state_dict())    

        time_elapsed = time.time() - since

        print('Training complete!!  in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))


        # ベストモデルの重みをロードします
        model.load_state_dict(best_model_state_dict)
        #  失敗した例を画像として保存
        num_images = 4
        visualize_model(device,output_dir,model,dataloaders,class_names,num_images)
        mlflow.log_artifact(f"{output_dir}/output.png")


        # mlflowにモデルを保存                            
        # 保存先のパス
        state_dict_path = "model_state_dict"
        mlflow.pytorch.log_state_dict(best_model_state_dict, state_dict_path)
        log.info(f"state_dict_uri: {mlflow.get_artifact_uri(state_dict_path)} \n ")

        ## モデルをmlflow
        # model_path = "model"
        # signature = infer_signature(inputs.numpy(),model(inputs).detach().numpy())
        # mlflow.pytorch.log_model(model, model_path, signature=signature,registered_model_name=cfg.mlflow.registered_model_name)

        # 実行logファイルをmlflowに保存        
        mlflow.log_artifact(f"{output_dir}/exec.log")


    mlflow.end_run()



if __name__ == "__main__":
    main()