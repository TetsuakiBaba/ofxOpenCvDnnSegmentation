#!/usr/bin/env bash
set -euxo pipefail
path_to_weights="https://tetsuakibaba.jp/tmp/enet-model-best.net"
if wget ${path_to_weights}; then
	echo "downloading Enet model"
else
	echo "wget is not installed. Falling back to curl"
	curl -O "${path_to_weights}"
fi
cp ./enet-model-best.net ./Examples/single_image_or_video/bin/data/dnn/
rm -f ./enet-model-best.net
echo "done"
