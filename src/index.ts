import express, { Express, Request, Response } from 'express';
import dotenv from 'dotenv';
import {ModelClient, Prediction} from "./client";
import multer from 'multer';

dotenv.config();

const PORT: number = parseInt(process.env.PORT || "3000");
const HOST: string = process.env.HOST || "127.0.0.1";
const MODEL_FP: string = process.env.MODEL_FP || "./resources/model/model.json";
const app: TensorExpress = express();
const upload = multer();
interface TensorExpress extends Express {
    model?: ModelClient
}

/**
 * GET /
 *
 * Returns "Server is working." if the server is online
 *
 * PARAMETERS:
 *   - None
 *
 * RESPONSE:
 *   {
 *     "message": "Server is working."
 *   }
 *
 */
app.get('/', (req: Request, res: Response): Response => {

    return res.json({
       "message": "Server is working."
    });

});


/**
 * POST /evaluate
 *
 * Evaluate result of an antigen test with a prediction of positivity/negativity and the associated confidences
 *
 * PARAMETERS:
 *   - Multipart file with "image" parameter containing an image Buffered File object.
 *
 * RESPONSE:
 *   - SUCCESSFUL PREDICTION
 *   {
 *       "message": "...",
 *       "status:" 200,
 *       "result: {
 *            "shape": [1, 2], // Shape of output tensor
 *            "data": [0.50, 0.50], // Tensor predictions on classes [NEGATIVE, POSITIVE]
 *            "positive": true // Whether the test is 'positive'
 *       }
 *   }
 *   - FILE MISSING
 *   {
 *      "message": "File missing.",
 *      "status": 400,
 *      "result": null
 *   }
 *  - ERROR
 *  {
 *      "message": "Internal error.",
 *      "status: 500,
 *      "result": null
 *  }
 *

 */
app.post("/evaluate", upload.single("image"), async (req: Request, res: Response): Promise<Response> => {

    // If file not provided
    if (!req.file) {
        return res.status(400).json({
            "message": "File missing.",
            "status": 400,
            "result": null
        });
    }

    // If model not loaded
    if (!app.model) {
        return res.status(500).json({
            "message": "Model failed load.",
            "status": 500,
            "result": null
        });
    }

    let prediction: Prediction;

    // Predict response
    try {
        const image = await app.model.convert(req.file.buffer);
        prediction = await app.model.evaluate(image);
        image.dispose()
    } catch (ex: any) {
        // Error returned
        return res.status(500).json({
            "message": "Internal error.",
            "status": 500,
            "result": null
        });
    }

    // Successful response
    return res.json({
        "message": "Successfully evaluated image.",
        "status": 200,
        "result": prediction
    })

});

/**
 * Load the model, then start the server.
 */
(async function () {

    app.model = await ModelClient.loadFromFiles(MODEL_FP);
    console.log("Successfully loaded model!");

    // Start HTTP server
    app.listen(PORT, HOST, async () => {
        console.log(`Server is running at http://${HOST}:${PORT}`);
    });

})()
