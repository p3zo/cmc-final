## Usage

Install [Docker Compose](https://docs.docker.com/compose/install).

#### Setup

Start the container and get a shell inside it

```bash
docker-compose up -d && docker-compose exec app bash
```

## Run EMOPIA Transformer

Download the npz files from [here](https://drive.google.com/file/d/17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP/view) and the pre-trained model state from [here](https://drive.google.com/u/0/uc?id=19Seq18b2JNzOamEQMG1uarKjj27HJkHu&export=download). Save them to `scripts/emopia_transformer`.

```bash
docker run -d -p 5000:5000 r8.im/annahung31/emopia@sha256:adeecd73e3c264c808710f4beb00c1affb130475f1e4848e1033e03894f8d38c

curl http://localhost:5000/predict -X POST -F emotion="low" -F seed=-1
```
