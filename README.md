# DecoMT: Decomposed Prompting for Machine Translation  
  
DecoMT simplifies the machine translation task between related languages by leveraging the monotonic alignment characteristic of such languages. DecoMT leverages few-shot prompting to decompose the translation process into a sequence of word chunk translations. We show that our approach surpasses multiple established few-shot baseline models, especially in scenarios involving low-resource languages  
  
## Contributions  
- Introduce Decomposed Prompting for MT (DecoMT), simplifying the translation task by dividing it into the translation of word chunks.  
- Perform extensive evaluations on closely related languages from diverse language families.  
- Compare DecoMT against several robust baselines, demonstrating its robust results, particularly outperforming in scenarios involving low-resource languages.  
  
## Usage  
Command to invoke decomposed prompting for machine translation:  
```  
MODEL_NAME="xl"
MODE="test"
BATCH_SIZE=5
SOURCE_LANGUAGE="mal_Mlym"
TARGET_LANGUAGE="hin_Deva"
CHUNK_SIZE=3
ROOT_OUTPUT_DIR=/data/output
prompt_template="Translate from Malayalam to Hindi:\nMalayalam: തിങ്കളാഴ്ച്ച, സ്റ്റാൻഫോർഡ് യൂണിവേഴ്‌സിറ്റി സ്‌കൂൾ\nHindi: सोमवार को, स्टैनफ़ोर्ड यूनिवर्सिटी स्कूल\nMalayalam: ഓഫ് മെഡിസിനിലെ ശാസ്ത്രജ്ഞന്മാർ\nHindi: ऑफ़ मेडिसिन के वैज्ञानिकों ने\nMalayalam: കോശങ്ങളെ അവയുടെ ഇനം\nHindi: कोशिकाओं को उनके प्रकार के\nMalayalam: അനുസരിച്ച് തരംതിരിക്കാൻ കഴിയുന്ന\nHindi: आधार पर छाँट सकने वाला\nMalayalam: ഒരു പുതിയ രോഗനിർണയ ഉപകരണം\nHindi: एक नए डायग्नोस्टिक उपकरण के\nMalayalam: കണ്ടുപിടിച്ചതായി പ്രഖ്യാപിച്ചു.\nHindi: आविष्कार की घोषणा की.\n\nTranslate from Malayalam to Hindi:\nMalayalam: മുന്‍നിര ഗവേഷകര്‍ പറയുന്നത്\nHindi: प्रसिद्ध शोधकर्ताओं ने कहा है कि\nMalayalam: ഇത് അര്‍ബുദം, ക്ഷയം, എച്ച്‍ഐ‍വി,\nHindi: यह कैंसर, टीवी, एचआईवी,\nMalayalam: മലേറിയ പോലുള്ള രോഗങ്ങളുടെ നേരത്തെയുള്ള\nHindi: मलेरिया जैसी बीमारियों का जल्द\nMalayalam: കണ്ടെത്തൽ സാധ്യമാക്കും\nHindi: पता लगाने में सक्षम हो\nMalayalam: എന്നാണ്. താഴ്ന്ന വരുമാനമുള്ള\nHindi: सकता है. अल्प आय वाले\nMalayalam: രാജ്യങ്ങളില്‍ സ്തനാര്‍ബുദം പോലുള്ള\nHindi: देशों में स्तन कैंसर जैसी\nMalayalam: രോഗങ്ങളില്‍ രോഗമുക്തി ലഭിക്കാനുള്ള\nHindi: बीमारियों से ठीक होने की\nMalayalam: സാദ്ധ്യത സമ്പന്ന രാജ്യങ്ങളുടെ പകുതി\nHindi: संभावना अमीर देशों से आधी\nMalayalam: മാത്രമാണ്.\nHindi: ही है.\n\nTranslate from Malayalam to Hindi:\nMalayalam: JAS 39C Gripen രാവിലെ ഏകദേശം\nHindi: JAS 39C Gripen सुबह करीब\nMalayalam: 9:30 ന്, പ്രാദേശിക സമയം (0230\nHindi: 9:30 बजे, स्थानीय समय (0230\nMalayalam: UTC) ക്ക് റൺവേയിലേക്ക് പൊട്ടിത്തെറിക്കുകയും\nHindi: UTC) को रनवे पर धमाके के साथ\nMalayalam: തകർന്നുവീഴുകയും ചെയ്തു,\nHindi: दुर्घटनाग्रस्त हो गया,\nMalayalam: അതിനാൽ എയർപോർട്ട്\nHindi: जिसकी वजह से हवाई अड्डे को\nMalayalam: കൊമേഴ്സ്യൽ വിമാനങ്ങൾക്കായി\nHindi: वाणिज्यिक उड़ानों के लिए\nMalayalam: അടച്ചിട്ടു.\nHindi: बंद कर दिया गया.\n\nTranslate from Malayalam to Hindi:\nMalayalam: പൈലറ്റെന്നാണ് തിരിച്ചറിഞ്ഞത് സ്ക്വാഡ്രൺ ലീഡർ\nHindi: पायलट की पहचान स्क्वाड्रन लीडर\nMalayalam: ദിലോക്രിത് പട്ടാവേ\nHindi: दिलोकृत पटावी के रूप में\nMalayalam: ആണ്.\nHindi: की गई.\n\nTranslate from Malayalam to Hindi:\nMalayalam: സംഭവ സ്ഥലത്തേക്ക് പോകുന്ന സമയത്ത്\nHindi: घटनास्थल की ओर जाते समय\nMalayalam: ഒരു എയർപോർട്ട് ഫയർ വാഹനം കീഴ്‌മേൽ മറിഞ്ഞതായി\nHindi: एक एयरपोर्ट अग्निशामक वाहन लुढ़क गई ऐसा\nMalayalam: പ്രാദേശിക മാധ്യമങ്ങൾ\nHindi: स्थानीय मीडिया ने\nMalayalam: റിപ്പോർട്ട് ചെയ്യുന്നു.\nHindi: बताया है.\n\nTranslate from Malayalam to Hindi:"
revision_template="Translate from Malayalam to Hindi:\nMalayalam: തിങ്കളാഴ്ച്ച, സ്റ്റാൻഫോർഡ് യൂണിവേഴ്‌സിറ്റി സ്‌കൂൾ ഓഫ് മെഡിസിനിലെ ശാസ്ത്രജ്ഞന്മാർ\nHindi: सोमवार को, स्टैनफ़ोर्ड यूनिवर्सिटी स्कूल ऑफ़ मेडिसिन के वैज्ञानिकों ने\nMalayalam: കോശങ്ങളെ അവയുടെ ഇനം അനുസരിച്ച് തരംതിരിക്കാൻ കഴിയുന്ന\nHindi: कोशिकाओं को उनके प्रकार के आधार पर छाँट सकने वाला\nMalayalam: ഒരു പുതിയ രോഗനിർണയ ഉപകരണം കണ്ടുപിടിച്ചതായി പ്രഖ്യാപിച്ചു.\nHindi: एक नए डायग्नोस्टिक उपकरण के आविष्कार की घोषणा की.\n\nTranslate from Malayalam to Hindi:\nMalayalam: മുന്‍നിര ഗവേഷകര്‍ പറയുന്നത് ഇത് അര്‍ബുദം, ക്ഷയം, എച്ച്‍ഐ‍വി,\nHindi: प्रसिद्ध शोधकर्ताओं ने कहा है कि यह कैंसर, टीवी, एचआईवी,\nMalayalam: മലേറിയ പോലുള്ള രോഗങ്ങളുടെ നേരത്തെയുള്ള കണ്ടെത്തൽ സാധ്യമാക്കും\nHindi: मलेरिया जैसी बीमारियों का जल्द पता लगाने में सक्षम हो\nMalayalam: എന്നാണ്. താഴ്ന്ന വരുമാനമുള്ള രാജ്യങ്ങളില്‍ സ്തനാര്‍ബുദം പോലുള്ള\nHindi: सकता है. अल्प आय वाले देशों में स्तन कैंसर जैसी\nMalayalam: രോഗങ്ങളില്‍ രോഗമുക്തി ലഭിക്കാനുള്ള സാദ്ധ്യത സമ്പന്ന രാജ്യങ്ങളുടെ പകുതി\nHindi: बीमारियों से ठीक होने की संभावना अमीर देशों से आधी\nMalayalam: മാത്രമാണ്.\nHindi: ही है.\n\nTranslate from Malayalam to Hindi:\nMalayalam: JAS 39C Gripen രാവിലെ ഏകദേശം 9:30 ന്, പ്രാദേശിക സമയം (0230\nHindi: JAS 39C Gripen सुबह करीब 9:30 बजे, स्थानीय समय (0230\nMalayalam: UTC) ക്ക് റൺവേയിലേക്ക് പൊട്ടിത്തെറിക്കുകയും തകർന്നുവീഴുകയും ചെയ്തു,\nHindi: UTC) को रनवे पर धमाके के साथ दुर्घटनाग्रस्त हो गया,\nMalayalam: അതിനാൽ എയർപോർട്ട് കൊമേഴ്സ്യൽ വിമാനങ്ങൾക്കായി\nHindi: जिसकी वजह से हवाई अड्डे को वाणिज्यिक उड़ानों के लिए\nMalayalam: അടച്ചിട്ടു.\nHindi: बंद कर दिया गया.\n\nTranslate from Malayalam to Hindi:\nMalayalam: പൈലറ്റെന്നാണ് തിരിച്ചറിഞ്ഞത് സ്ക്വാഡ്രൺ ലീഡർ ദിലോക്രിത് പട്ടാവേ\nHindi: पायलट की पहचान स्क्वाड्रन लीडर दिलोकृत पटावी के रूप में\nMalayalam: ആണ്.\nHindi: की गई.\n\nTranslate from Malayalam to Hindi:\nMalayalam: സംഭവ സ്ഥലത്തേക്ക് പോകുന്ന സമയത്ത് ഒരു എയർപോർട്ട് ഫയർ വാഹനം കീഴ്‌മേൽ മറിഞ്ഞതായി\nHindi: घटनास्थल की ओर जाते समय एक एयरपोर्ट अग्निशामक वाहन लुढ़क गई ऐसा\nMalayalam: പ്രാദേശിക മാധ്യമങ്ങൾ റിപ്പോർട്ട് ചെയ്യുന്നു.\nHindi: स्थानीय मीडिया ने बताया है.\n\nTranslate from Malayalam to Hindi:"  
python script/decomt.py --mode $MODE --source_language $SOURCE_LANGUAGE \
--target_language $TARGET_LANGUAGE --prompt_template ${prompt_template} \
--revision_template ${revision_template}  --batch_size $BATCH_SIZE --chunk_size ${CHUNK_SIZE} \
--root_output_directory ${ROOT_OUTPUT_DIR} \
--model_name ${MODEL_NAME}
```  
  
Command to invoke baseline models:  
```  
MODEL_NAME="xl"
MODE="test"
BATCH_SIZE=5
SOURCE_LANG="mal_Mlym"
TARGET_LANG="hin_Deva"
prompt_template="Translate from Malayalam to Hindi:\nMalayalam: തിങ്കളാഴ്ച്ച, സ്റ്റാൻഫോർഡ് യൂണിവേഴ്‌സിറ്റി സ്‌കൂൾ ഓഫ് മെഡിസിനിലെ ശാസ്ത്രജ്ഞന്മാർ കോശങ്ങളെ അവയുടെ ഇനം അനുസരിച്ച് തരംതിരിക്കാൻ കഴിയുന്ന ഒരു പുതിയ രോഗനിർണയ ഉപകരണം കണ്ടുപിടിച്ചതായി പ്രഖ്യാപിച്ചു: സ്റ്റാൻഡേർഡ് ഇങ്ക്‌ജെറ്റ് പ്രിന്റ്ററുകൾ ഉപയോഗിച്ച് നിർമ്മിക്കാൻ സാധിക്കുന്ന ഏകദേശം ഒരു യു.എസ് സെന്റ് ഓരോന്നിനും വേണ്ടിവരുന്ന പ്രിന്റ് ചെയ്യാൻ കഴിയുന്ന ഒരു ചെറിയ ചിപ്പ്.\nHindi: सोमवार को, स्टैनफ़ोर्ड यूनिवर्सिटी स्कूल ऑफ़ मेडिसिन के वैज्ञानिकों ने एक नए डायग्नोस्टिक उपकरण के आविष्कार की घोषणा की जो कोशिकाओं को उनके प्रकार के आधार पर छाँट सकता है: एक छोटी प्रिंट करने योग्य चिप जिसे स्टैण्डर्ड इंकजेट प्रिंटर का उपयोग करके लगभग एक अमेरिकी सेंट के लिए निर्मित किया जा सकता है.\n\nTranslate from Malayalam to Hindi:"  
 
python script/sap.py --mode $MODE --source_language $SOURCE_LANGUAGE \
--target_language $TARGET_LANGUAGE --prompt_template ${prompt_template} \
--batch_size $BATCH_SIZE --root_output_directory ${ROOT_OUTPUT_DIR} \
--model_name ${MODEL_NAME}
  
python script/sp-mt5.py --mode $MODE --source_language $SOURCE_LANGUAGE \
--target_language $TARGET_LANGUAGE --prompt_template ${prompt_template} \
--batch_size $BATCH_SIZE --root_output_directory ${ROOT_OUTPUT_DIR} \
--model_name ${MODEL_NAME}
  
MODEL_NAME="bigscience/bloom-7b1" or MODEL_NAME="facebook/xglm-7.5B"
python script/sp.py --mode $MODE --source_language $SOURCE_LANGUAGE \
--target_language $TARGET_LANGUAGE --prompt_template ${prompt_template} \
--batch_size $BATCH_SIZE --root_output_directory ${ROOT_OUTPUT_DIR} \
--model_name ${MODEL_NAME}
```  
   
## Installation
```
pip install torch==1.13.1  
pip install transformers==4.34.0  
pip install sacrebleu  
```

## Downloads
  Download the model outputs from https://drive.google.com/file/d/1ZexXHSHy-_GlAA5kJzSYRJawkG4cVfiX/view?usp=sharing 

## Citation  
If you find this code useful for your research, please cite the following paper:  
```  
@misc{puduppully2023decomt,  
      title={Decomposed Prompting for Machine Translation Between Related Languages using Large Language Models},   
      author={Ratish Puduppully and Anoop Kunchukuttan and Raj Dabre and Ai Ti Aw and Nancy F. Chen},  
      year={2023},  
      eprint={2305.13085},  
      archivePrefix={arXiv},  
      primaryClass={cs.CL}  
}  
```
