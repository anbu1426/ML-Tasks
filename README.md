# ML Models
Three deep learning models created with PyTorch

## Models
## Model - 1 -> BaseTaskModel(FastionMNISTModel)
**Brief Description** ->
- Created a multi-class classification model using Linear Functions as well as Convutional Neutral Networks(CNNs).
- On testing we created two versions of model V0 and V1, in which V0 uses Linear Functions for classification while V1 uses CNNs.
- V1 version is superior to V0 having more than 9% higher accuracy than V0.
- Loss v/s epochs plots are also plotted.

**Dataset** -> We used FashionMNIST dataset from `torchvision.datasets`

**Architecture** -> We used TingVGG like architecture

## Model - 2 -> StreamTaskModel(RAG_Chatbot)
**Brief Description** -> 
- A Retrieval Augmented Generation chatbot that answers question from uploaded PDFs.
- Upload any PDF and ask questions about it in a conversatinal way.
- Maintains the chat-history so follow-up questions works.
- Runs completely offline as also run by API key tokens too.

**Dataset** -> Any data in form of PDF(mainly)

**Architecture** ->
- PDF text extaction via PyPDF2.
- Text chunking via LangChain CharacterTextSplitter.
- Emdeddings via HuggingFace(all-MiniLM-L6-v2)
- Vector store via FAISS
- LLM via Olama (phi3) - runs locally but can also be used with API tokens
- Memory of chatbot via RunnableWithMessageHistory
- UI via Streamlit

## Model - 2 -> BaseTask's-BonusTask(ImageEncodeDecodeModel)
**Brief Description** ->
- A ImageEncodeDecode Model makes the given image undergo to processes:
  1. Encoding of the Image(Compressing the image) -> We used `Conv2d()`, `ReLU()`, and `MaxPool2d()` fns for compressing the image.
     In which input shape was 1x28x28 -> hidden_unitsx28x28 ...(Compressing)-> Final output shape was hidden_unitsx7x7.
  2. Decoding of the Image(Reconstructing the compressed image) -> We use `ConvTranspose2d()`, `ReLU()`, and finally `Sigmoid()` activation function for reconstructing the image.
     In which for this input shape is hidden_unitsx7x7 ...(Reconstructed)-> Final output shape is 1x28x28.
- Though we retained the orginal shape but not the same quality of the image.
- To increase the quality we created a training loop and Increased the accuracy of the compression and reconstruction of the image.

**Dataset** -> We used FashionMNIST dataset from `torchvision.datasets`

**Architecture** ->
- Created the training and testing data.
- Created the model class with encode and decode part.
- Tested with a sample testdata and got to know the model is too bad at compressing and reconstructing the image.
- So we used loss fn as `MSELoss()` from nn module and for optimizer we use `torch.optim.Adam()`.
- We converting the training data into train dataloader.
- Created the training loop and trained with train dataloader.
- By experimenting we found that hidden_units of 16, 32 where not good for the model as it gave loss of 0.2 or 0.1, so I tried with hidden_units = 64 and found that very good result i.e., loss is about 0.003.
- Tested with the testing data and found very good results.
- Now the model is capable of compressing and reconstructing image with minimum errors and data losses!. 

