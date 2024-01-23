mkdir checkpoints

# if checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt exists
if [ -f checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt ]; then
    echo "checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt exists"
else
    echo "Downloading checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_aa
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ab
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ac
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ad
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ae
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_af
    
    cat v1-5-pruned-emaonly-adaption-task-humanalign_* > checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt
    rm v1-5-pruned-emaonly-adaption-task-humanalign_*
fi

if [ -f checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt ]; then
    echo "checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt exists"
else
    echo "Downloading checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt"
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_aa
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ab
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ac
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ad
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ae
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_af

    cat v1-5-pruned-emaonly-adaption-task_* > checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt
    rm v1-5-pruned-emaonly-adaption-task_*
fi

# download ip2p from http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt
if [ -f checkpoints/instruct-pix2pix-00-22000.ckpt ]; then
    echo "checkpoints/instruct-pix2pix-00-22000.ckpt exists"
else
    echo "Downloading checkpoints/instruct-pix2pix-00-22000.ckpt"
    wget http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt -P checkpoints/
fi

# download magicbrush from wget https://huggingface.co/osunlp/InstructPix2Pix-MagicBrush/resolve/main/MagicBrush-epoch-52-step-4999.ckpt
if [ -f checkpoints/MagicBrush-epoch-52-step-4999.ckpt ]; then
    echo "checkpoints/MagicBrush-epoch-52-step-4999.ckpt exists"
else
    echo "Downloading checkpoints/MagicBrush-epoch-52-step-4999.ckpt"
    wget https://huggingface.co/osunlp/InstructPix2Pix-MagicBrush/resolve/main/MagicBrush-epoch-52-step-4999.ckpt -P checkpoints/
fi

# checkpoint list
# checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt
# checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt
# checkpoints/instruct-pix2pix-00-22000.ckpt
# checkpoints/MagicBrush-epoch-52-step-4999.ckpt
