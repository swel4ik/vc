## Overview
```pre
├── inference_pipeline.py
├── model_serving.py
├── README.md
├── server_config.yaml
├── server.py
├── test_server.py
└── utils.py
```
### server_config.yaml
Specifies paths to the models and general parameters(batch size, inference device)

### Run
```
python server.py --ip <ip_address> --port <port> --logfile <path to log file>
```
### Test
```
python test_server.py --ip <ip_address> --port <port> --text <json file with text to be generated> --voice <voice .wav file> --output <path to output folder> --format <Extension for output audio files ("wav" or "flac")> --out_structure <he structure how to process the output response ("zip" or "multipart")>
```
### Endpoint
The endpoint accepts POST requests to the address: `http://<ip>:<port>/api/inference`
### Requirements for request files:  
You can see the examples in `post_request_files` directory     
1. Value `text` - `.json` file with following structure:  
```
{
"format": "wav",
"out": "zip",
"texts": {
    "Text_1": "Hello, I am a text to speech model from the Brouton Lab. They were also asked to discuss any changes planned for future implementation or changes they would like to implement.",
    "Text_2": "Good luck!"
        }
}
```
`"format"` - extension for output audio files. Supported: `.wav`, `.flac`.  
`"out"` - type of the response for client side. Supported: `zip`, `MultipartEncoder`.  
`"texts"` - list of texts for generating.  
2. Value `voice` - `.wav` file with duration from 3.5 to 10 seconds. 
