# [Deprecated] Tools - Web App

> [!WARNING]  
> This branch contains the deprecated tools web application. To see the latest CLI version, please refer to the [main](https://github.com/luxonis/tools/tree/main) branch.

This application is used for exporting Yolo V5, V6 and V7 object detection models for OAKs.

## Running the app locally
To run the application locally you need to do these folllowing steps.
### Step 1: Cloning the tools repository and all submodules
```
git clone --recursive https://github.com/luxonis/tools.git
```

### Step 2: Updating the `nginx.conf` file
It is required for the app to not consider SSL. You can rewrite the current `nginx.conf` with the following file: 
```
upstream toolsapi {
    server api:8000;
}

upstream yolov7 {
    server yolov7:8001;
}

upstream yolov6r1 {
    server yolov6r1:8002;
}

upstream yolov6r3 {
    server yolov6r3:8003;
}

server {
  listen 80;
  listen [::]:80;
  client_max_body_size 1G;
  access_log                /log/access.unsecured.log;
  error_log                 /log/error.unsecured.log error;
  proxy_read_timeout 1h;

  location /yolov7 {
    proxy_pass http://yolov7;
    proxy_redirect off;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Ssl $scheme; 
  }

  location /yolov6r1 {
    proxy_pass http://yolov6r1;
    proxy_redirect off;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Ssl $scheme; 
  }

  location /yolov6r3 {
    proxy_pass http://yolov6r3;
    proxy_redirect off;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Ssl $scheme; 
  }

  location / {
    proxy_pass http://toolsapi;
    proxy_redirect off;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Ssl $scheme;
  }
}
```

**Note: It's better to create a copy of the `nginx.conf` file before rewriting it.**

### Step 3: Updating the `docker-compose.yml` file
The app must be built from local images. You can do it by updating the current `docker-compose.yml` with the following file:
```
version: '2'

services:
  api:
    build: .
    image: tools/api:latest
    ports:
      - 8000:8000
  yolov7:
    build: ./yolov7
    image: tools/yolov7:latest
    ports:
      - 8001:8001
  yolov6r1:
    build: ./yolov6r1
    image: tools/yolov6r1:latest
    ports:
      - 8002:8002
  yolov6r3:
    build: ./yolov6r3
    image: tools/yolov6r3:latest
    ports:
      - 8003:8003
  nginx:
    build: ./nginx
    image: tools/nginx:latest
    ports:
      - 8080:80
      - 8443:443
    depends_on:
      - api
      - yolov7
      - yolov6r1
      - yolov6r3
    volumes:
      - ./log:/log:z
      - /etc/letsencrypt/live/tools.luxonis.com/privkey.pem:/ssl/key.pem
      - /etc/letsencrypt/live/tools.luxonis.com/fullchain.pem:/ssl/cert.pem
```

**Note: It's better to create a copy of the `docker-compose.yml` file before rewriting it.**

### Step 4: Building
```sudo docker-compose build```
or
```sudo docker compose build```

### Step 5: Running
```sudo docker-compose up```
or
```sudo docker compose up```

### Step 6: Open browser
Open browser at [http://localhost:8080/](http://localhost:8080/).

## Credits

This application uses source code of the following repositories: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv7](https://github.com/WongKinYiu/yolov7), and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (see each of them for more information).

## License

This application is available under **AGPL-3.0 License** license (see [LICENSE](https://github.com/luxonis/tools/blob/master/LICENSE) file for details).
