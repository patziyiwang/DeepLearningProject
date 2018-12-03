# DeepLearningProject
Object dynamics prediction with attention

To run the code, first run preprocess.py to convert text file annotation to
maps. Changes need to be made to load box dimensions correctly.

After data is preprocessed, run train_nn.py, which has two training modes: batch
and recurrent. Batch training always feeds in the ground truth and predicts the
next frame. Recurrent training gives a certain number of ground truth frames
(seen_step) and predicts the next couple of frames (fut_step). Finally, the model
can be evaluated using mode_eval.

Batch training works fine now, but train_recurrent and mode_eval might need some
debugging. Finally, we also need some code to visualize the output to better
compare against the ground truth.
