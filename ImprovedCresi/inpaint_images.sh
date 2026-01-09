for file in data/img_*.jpeg; do
    python3 main.py --num-inference-steps=20 --tile-size=512 --image-local=$file --output-path="inpainted_$(basename $file).png"
done
