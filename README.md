## Usage

Install [Docker Compose](https://docs.docker.com/compose/install).

#### Setup

Start the container and get a shell inside it

```bash
docker-compose up -d && docker-compose exec app bash
```

### Run the EMOPIA transformer

Download the npz files from [here](https://drive.google.com/file/d/17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP/view) and the pre-trained model state from [here](https://drive.google.com/u/0/uc?id=19Seq18b2JNzOamEQMG1uarKjj27HJkHu&export=download). Save them to `scripts/emopia_transformer`.

```bash
pip install -r requirements.txt
```

Run `scripts/generate.py` to generate 10 midi files for each emotion class to `output/emopia-output`. This script is also available as a [Colab Notebook](https://colab.research.google.com/drive/1ZWHUZJA09cmnOpzpUMfAw06NO7cchbvY?usp=sharing).

### Extract chords from midi

Run `scripts/extract_chords.py`.
