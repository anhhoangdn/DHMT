.PHONY: setup landmark deca pipeline clean

INPUT ?= data/samples/test.jpg
OUTPUT ?= outputs
DECA_ROOT ?= external/DECA

setup:
	bash setup.sh

landmark:
	python src/preprocess/landmark_mediapipe.py --input $(INPUT) --output_dir $(OUTPUT)/landmarks2d

deca:
	python src/recon/run_deca.py --input $(INPUT) --output $(OUTPUT) --deca-root $(DECA_ROOT)

pipeline:
	python src/pipeline/run_pipeline.py --input $(INPUT) --output $(OUTPUT) --deca-root $(DECA_ROOT)

clean:
	rm -rf outputs/landmarks2d/* outputs/meshes/* outputs/textures/* outputs/renders/* outputs/videos/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
