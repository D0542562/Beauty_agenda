# beauty-agenda Service


## Installation

1. Change working directory to this folder
2. Build docker image
    ```bash
    docker build -t shes990129/beauty-agenda .
    ```

## Start service

1. Start the service using `docker-compose`
    ```
    docker-compose up -d
    ```
2. Service is now on port `5000`  
    Note. The service listens only localhost, if you want it listen all interfaces instead of localhost only, please change `127.0.0.1:5000:5000` to `5000:5000` in `docker-compose.yml` and restart the service. Or you can use reverse proxy of HTTP server.
3. the best
    ```
     docker-compose down && docker-compose up -d && docker-compose logs -f
    ```

## Stop service

1. Stop the service using `docker-compose`
    ```
    docker-compose down
    ```

---

## Endpoint

- Main
  - method: `POST`
  - route: `/`
  - parameter
    - `image`: File. image


## Test service

1. Send request using curl
    ```bash
    curl -X POST localhost:5000 -F "image=@path/to/078f32485fee528371e68caedc64a0a6-1-13243-1.jpg"
    ```
2. Get the response like the following one
    ```json
    {
        
    }
    ```

## Alternative

If you want to run this service without Docker, you can follow this steps after `data` folder is ready.  
However, **we highly recommend using Docker Compose**.

1. Install required packages
    ```bash
    pip3 install -r requirements.txt
    ```
2. Run
    ```bash
    python3 entrypoint.py
    ```
