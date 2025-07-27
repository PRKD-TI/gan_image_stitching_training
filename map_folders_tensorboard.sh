#!/bin/bash

# Desmonta diretórios (ignora erro se já estiverem desmontados)
echo "Desmontando ./datasetzip..."
sudo fusermount -u ./datasetzip || true

echo "Desmontando ./logs..."
sudo fusermount -u ./logs || true

echo "Desmontando ./checkpoints_epoch..."
sudo fusermount -u ./checkpoints_epoch || true

echo "Desmontando ./checkpoints_batch..."
sudo fusermount -u ./checkpoints_batch || true

echo "Desmontando ./utils..."
sudo fusermount -u ./utils || true

# Remove e recria diretórios locais
echo "Removendo diretórios locais..."
rm -rf ./datasetzip ./logs ./checkpoints_epoch ./checkpoints_batch ./utils || true

echo "Criando diretórios locais..."
mkdir -p ./datasetzip ./logs ./checkpoints_epoch ./checkpoints_batch ./utils

# Opções SSHFS comuns
SSHFS_OPTS="-o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,allow_other -o IdentityFile=/home/prkd/.ssh/colab_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"  # reconexão automática

# Monta com sshfs
echo "Montando ./datasetzip..."
sshfs $SSHFS_OPTS prkdvps@64.71.153.122:/home/prkdvps/datasetzip ./datasetzip

echo "Montando ./logs..."
sshfs $SSHFS_OPTS prkdvps@64.71.153.122:/home/prkdvps/tensorboard/logs ./logs

echo "Montando ./checkpoints_epoch..."
sshfs $SSHFS_OPTS prkdvps@64.71.153.122:/home/prkdvps/tensorboard/checkpoints_epoch ./checkpoints_epoch

echo "Montando ./checkpoints_batch..."
sshfs $SSHFS_OPTS prkdvps@64.71.153.122:/home/prkdvps/tensorboard/checkpoints_batch ./checkpoints_batch

echo "Montando ./utils..."
sshfs $SSHFS_OPTS prkdvps@64.71.153.122:/home/prkdvps/utils ./utils

echo "✅ Tudo montado com sucesso."
