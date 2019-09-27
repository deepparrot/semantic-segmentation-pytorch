echo "Moving files"
mv ./data/validation.odgt ./.data/vision/ade20k/validation.odgt
mv ./data/training.odgt ./.data/vision/ade20k/training.odgt
echo "Files moved"

apt-get install unzip
unzip ./.data/vision/ade20k/ADEChallengeData2016.zip -d ./.data/vision/ade20k/
echo "Dataset downloaded."
