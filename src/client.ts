import {LayersModel, Tensor} from "@tensorflow/tfjs-node";
import jpeg, {BufferRet} from "jpeg-js";

const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const sharp = require("sharp");

/**
 * Output of prediction from the model
 */
export interface Prediction {
    shape: number[]
    data: number[],
    positive: boolean
}

/**
 * Client to read in images and check their positivity
 */
export class ModelClient {

    private readonly model: LayersModel;
    private readonly inputHeight: number = 256;
    private readonly inputWidth: number = 256;

    constructor(model: LayersModel) {
        this.model = model;
    }

    /**
     * Load the client from a TensorFlow model.json file
     * @param fp File path of the file
     */
    static async loadFromFiles(fp: string) {
        const loadHandler = tfn.io.fileSystem(fp);
        const model = await tf.loadLayersModel(loadHandler);
        return new this(model);
    }

    /**
     * Load an image from a Buffer as a "convertable" image (standardized).
     * @param img The image buffer to convert
     */
    private async loadImage(img: Buffer): Promise<BufferRet> {

        // Convert ALL images to 256x256 jpeg
        const fromBuffer: Buffer = await sharp(img)
            .resize(this.inputWidth, this.inputHeight, {
                fit: "fill" // Stretches image (similar to how we trained it)
            })
            .grayscale(true)
            .jpeg({mozjpeg: true})
            .toBuffer();

        // Load as processable jpeg
        return jpeg.decode(fromBuffer);

    }

    /**
     * Convert an image buffer of any type (jpeg, png, etc.) to a Tensor
     * @param img The buffered image
     */
    async convert(img: Buffer): Promise<Tensor> {

        const image = await this.loadImage(img);

        const numChannels = 1;
        const numPixels = image.width * image.height;
        const values = new Int32Array(numPixels * numChannels);

        for (let i = 0; i < numPixels; i++)
            for (let c = 0; c < numChannels; ++c)
                values[i * numChannels + c] = image.data[i * 4 + c]

        return tf.tensor(values, [1, image.height, image.width, numChannels], 'int32');
    }

    /**
     * Evaluate a Tensor representation of an image against the API
     * @param image A Tensor containing image data
     */
    async evaluate(image: Tensor): Promise<Prediction> {

        // Get result tensor
        const tensor = this.model.predict(image) as Tensor;

        // Extract data, [0.123123, 0.1231231]
        // The array shape is [% likelihood negative, % likelihood positive]
        const data = Array.from(await tensor.data());

        // Return results
        return (
            {
                shape: tensor.shape,
                data: data,
                positive: this.parsePositive(data)
            }
        )

    }

    private parsePositive(result: number[]): boolean {
        return result[1] > 0.95;
    }

}
