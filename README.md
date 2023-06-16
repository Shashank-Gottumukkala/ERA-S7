# MNSIT Digits using PyTorch
### The following repository contains files that have been tested to try to achieve an accuracy of `99.4%` in more than one epoch, while keeping the model under `8k` parameters

## Code Iterations:
-  ## `S7- Model 1.ipynb` :
   ## Target:
      - Get the Setup Right
      - Get a pretty decent accuracy, don't worry about number of params right now
      - Make sure the data loaders are working
      - Setup the basic training loop
      - Create a Baseline Model
      - 15 Epoch

    ## Result:
      - Parameters : `25,648`
      - Best Training Accuracy : `99.61`
      - Best Test Accuracy : `99.07`

    ## Analysis:
      - The model is a bit large compared to our target model of having under `8k` parameters
      - The Model is Overfitting
          
   ## `S7- Model 2.ipynb` :
   -    ## Target:
          - Reduce the number of parameters
          - Add regularization techniques like Batch Normalization and Dropout to prevent overfitting

        ## Result:
          - Parameters : `12,100`
          - Best Training Accuracy : `99.15`
          - Best Test Accuracy : `99.11`

        ## Analysis:
          - The model is not overfitting, but we need to reduce the number of params, since our target is to make the model under `8k` Params
          - The Model seems to be generalizing well
  - ##`S7- Model 3.ipynb` : 
  - ##`S7- Model 4.ipynb` :
  - ##`S7- Model 5.ipynb` :  
