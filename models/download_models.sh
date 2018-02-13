cd models

ROOT_URL=https://lmb.informatik.uni-freiburg.de/data/fast-artistic-videos/models

# Video models
wget $ROOT_URL/checkpoint-mosaic-video.t7
wget $ROOT_URL/checkpoint-picasso-video.t7
wget $ROOT_URL/checkpoint-schlief-video.t7
wget $ROOT_URL/checkpoint-scream-video.t7
wget $ROOT_URL/checkpoint-WomanHat-video.t7
wget $ROOT_URL/checkpoint-candy-video.t7
# Image models
wget $ROOT_URL/checkpoint-mosaic-image.t7
wget $ROOT_URL/checkpoint-picasso-image.t7
wget $ROOT_URL/checkpoint-schlief-image.t7
wget $ROOT_URL/checkpoint-scream-image.t7
wget $ROOT_URL/checkpoint-WomanHat-image.t7
wget $ROOT_URL/checkpoint-candy-image.t7
cd ..
