# from transformers.models.bert.tokenization_bert import BertTokenizer

# class Tokenizer(BertTokenizer):
#     def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]], add_special_tokens: bool, padding: Union[bool, str, PaddingStrategy], truncation: Union[bool, str, TruncationStrategy], max_length: Optional[int], stride: int, return_tensors: Optional[Union[str, TensorType]], **kwargs) -> List[int]:
#         return super().encode(text, text_pair=text_pair, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, return_tensors=return_tensors, **kwargs)


#     def decode(self, token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"], skip_special_tokens: bool, clean_up_tokenization_spaces: bool, **kwargs) -> str:
#         return super().decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)