"""Example of how to add a model in tape.

This file shows an example of how to add a new model to the tape training
pipeline. tape models follow the huggingface API and so require:

    - A config class
    - An abstract model class
    - A model class to output sequence and pooled embeddings
    - Task-specific classes for each individual task

This will walkthrough how to create each of these, with a task-specific class for
secondary structure prediction. You can look at the other task-specific classes
defined in e.g. tape/models/modeling_bert.py for examples on how to
define these other task-specific models for e.g. contact prediction or fluorescence
prediction.

In addition to defining these models, this shows how to register the model to
tape so that you can use the same training machinery to run your tasks.
"""

import torch
import torch.nn as nn
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.registry import registry

from unirep import UniRep

class UniRepReimpConfig(ProteinConfig):
    """ The config class for our new model. This should be a subclass of
        ProteinConfig. It's a very straightforward definition, which just
        accepts the arguments that you would like the model to take in
        and assigns them to the class.

        Note - if you do not initialize using a model config file, you
        must provide defaults for all arguments.
    """

    def __init__(self, rnn_type: str = "LSTM", embed_size: int = 10, hidden_size: int = 1024, num_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.rnn_type = rnn_type
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initializer_range = 0.02

class UniRepReimpAbstractModel(ProteinModel):
    """ All your models will inherit from this one - it's used to define the
        config_class of the model set and also to define the base_model_prefix.
        This is used to allow easy loading/saving into different models.
    """
    config_class = UniRepReimpConfig
    base_model_prefix = 'unirep_reimp'

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@registry.register_task_model('embed', 'unirep_reimp')
class UniRepReimpModel(UniRepReimpAbstractModel):
    """ The base model class. This will return embeddings of the input amino
        acid sequence. It is not used for any specific task - you'll have to
        define task-specific models further on. Note that there is a little
        more machinery in the models we define, but this is a stripped down
        version that should give you what you need
    """
    # init expects only a single argument - the config
    def __init__(self, config: UniRepReimpConfig):
        super().__init__(config)
        self.inner_model = UniRep(config.rnn_type, config.embed_size, config.hidden_size, config.num_layers)
        self.init_weights()

    def forward(self, input_ids, input_mask = None):
        """ Runs the forward model pass

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid

        Returns:
            sequence_embedding (Tensor[float]):
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        embed = self.inner_model.embed(input_ids)

        if self.inner_model.rnn_type == "mLSTM":
            out, state = self.inner_model.rnn(embed, None, input_mask)
        else:
            out, state = self.inner_model.rnn(embed)

        if self.inner_model.rnn_type != "GRU":
            state = (state[0].squeeze(), state[1].squeeze())
            pooled_out = torch.cat(state, 1)
        else:
            pooled_out = state.squeeze()

        return (out, pooled_out)

# @registry.register_task_model('language_modeling', 'unirep_reimp')
# class UniRepReimpForLM(UniRepReimpAbstractModel):
#     # TODO: Fix this for UniRep - UniRep changes the size of the targets

#     def __init__(self, config):
#         super().__init__(config)

#         self.unirep = UniRepModel(config)
#         self.feedforward = nn.Linear(config.hidden_size, config.vocab_size - 1)

#         self.init_weights()

#     def forward(self,
#                 input_ids,
#                 input_mask=None,
#                 targets=None):

#         outputs = self.unirep(input_ids, input_mask=input_mask)

#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores = self.feedforward(sequence_output)

#         # add hidden states and if they are here
#         outputs = (prediction_scores,) + outputs[2:]

#         if targets is not None:
#             targets = targets[:, 1:]
#             prediction_scores = prediction_scores[:, :-1]
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
#             lm_loss = loss_fct(
#                 prediction_scores.view(-1, self.config.vocab_size), targets.view(-1))
#             outputs = (lm_loss,) + outputs

#         # (loss), prediction_scores, (hidden_states)
#         return outputs


@registry.register_task_model('fluorescence', 'unirep_reimp')
@registry.register_task_model('stability', 'unirep_reimp')
class UniRepReimpForValuePrediction(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)

        if self.unirep_reimp.inner_model.rnn_type == "GRU":
            predict_size = config.hidden_size
        else:
            predict_size = config.hidden_size * 2

        self.predict = ValuePredictionHead(predict_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('remote_homology', 'unirep_reimp')
class UniRepReimpForSequenceClassification(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)

        if self.unirep_reimp.inner_model.rnn_type == "GRU":
            predict_size = config.hidden_size
        else:
            predict_size = config.hidden_size * 2

        self.classify = SequenceClassificationHead(predict_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('secondary_structure', 'unirep_reimp')
class UniRepReimpForSequenceToSequenceClassification(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)
        self.classify = SequenceToSequenceClassificationHead(config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('contact_prediction', 'unirep_reimp')
class UniRepReimpForContactPrediction(UniRepReimpAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep_reimp = UniRepReimpModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_reimp(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

if __name__ == '__main__':
    """ To actually run the model, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, `run_eval`,
    and `run_embed`. Alternatively, you can simply place this file inside
    the `tape/models` directory, where it will be auto-imported
    into tape.
    """
    from tape.main import run_train, run_eval, run_embed
    run_train()
