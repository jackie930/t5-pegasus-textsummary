import tf2onnx
from bert4keras.models import build_transformer_model
from onnxruntime_tools import optimizer
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

if __name__ == '__main__':
    # bert4keras版本
    config_path = './chinese_t5_pegasus_base/config.json'
    checkpoint_path = './chinese_t5_pegasus_base/model.ckpt'
    dict_path = './chinese_t5_pegasus_base/vocab.txt'
    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='t5.1.1',
        return_keras_model=False,
        name='T5',
    )
    encoder = t5.encoder
    decoder = t5.decoder

    # convert tensorflow to onnx
    decoder_fp32 = './chinese_t5_pegasus_base/t5_decoder_fp32.onnx'
    encoder_fp32 = './chinese_t5_pegasus_base/t5_encoder_fp32.onnx'
    model_proto_decoder, tensor_storage_decoder = tf2onnx.convert.from_keras(
        decoder,
        output_path=decoder_fp32,
        opset=12)
    model_proto_encoder, tensor_storage_encoder = tf2onnx.convert.from_keras(
        encoder,
        output_path=encoder_fp32,
        opset=12)

    # # optimize models to float16，only support some models,details can be found in README.md
    # decoder_fp16 = './chinese_t5_pegasus_base/t5_decoder_fp16.onnx'
    # encoder_fp16 = './chinese_t5_pegasus_base/t5_encoder_fp16.onnx'
    #
    # decoder_optimized_model = optimizer.optimize_model(decoder_fp32, model_type='bert', num_heads=12, hidden_size=768)Ls19950124
    # decoder_optimized_model.convert_float_to_float16()
    # decoder_optimized_model.save_model_to_file(decoder_fp16)
    # encoder_optimized_model = optimizer.optimize_model(encoder_fp32, model_type='bert', num_heads=12, hidden_size=768)
    # encoder_optimized_model.convert_float_to_float16()
    # encoder_optimized_model.save_model_to_file(encoder_fp16)

    # dynamic quantize
    decoder_dynamic = './chinese_t5_pegasus_base/t5_decoder_dynamic.onnx'
    encoder_dynamic = './chinese_t5_pegasus_base/t5_encoder_dynamic.onnx'
    decoder_dynamic_model = quantize_dynamic(decoder_fp32, decoder_dynamic, activation_type=QuantType.QUInt8,
                                             weight_type=QuantType.QInt8)
    encoder_dynamic_model = quantize_dynamic(encoder_fp32, encoder_dynamic, activation_type=QuantType.QUInt8,
                                             weight_type=QuantType.QInt8)







