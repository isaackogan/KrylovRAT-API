# KrylovRAT-API

### Running Project

1. `npm install`
2. `npm run dev`


### Sending Request

- Send a POST request to `#/evaluate` endpoint
- Include a multipart form with an `image` field
- Set the `image` field to a `JPEG/PNG` image (streamed file)
- Set the `Content-Type` header to `multipart/formdata`

### Interpreting Response

- Payload will ALWAYS be a JSON response
- If successful, a "data" object will exist in the return JSON payload
- Prediction is a boolean value located in `<response>.data.positive`