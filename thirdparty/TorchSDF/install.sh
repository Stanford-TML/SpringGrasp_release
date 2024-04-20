if [ -d "build" ];then
    bash clean.sh
fi

python setup.py develop
python -c "import torchsdf; print(torchsdf.__version__)"