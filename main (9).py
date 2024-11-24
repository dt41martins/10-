  from transformers import MarianMTModel, MarianTokenizer

  model_name = "Helsinki-NLP/opus-mt-lv-en"
  model = MarianMTModel.from_pretrained(model_name)
  tokenizer = MarianTokenizer.from_pretrained(model_name)

  texts = [
      "Labdien! Kā jums klājas?",
      "Es šodien lasīju interesantu grāmatu."
  ]

  translated_texts = []

  for text in texts:
      translated = tokenizer.encode(text, return_tensors="pt", padding=True)
      translated_output = model.generate(translated, max_length=100)
      translated_text = tokenizer.decode(translated_output[0], skip_special_tokens=True)
      translated_texts.append(translated_text)

  for i, translated in enumerate(translated_texts):
      print(f"Original: {texts[i]}")
      print(f"Tulkojums: {translated}")
      print()
