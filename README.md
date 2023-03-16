# deep-learning-project-config
Repo to try out ideas on how to config a CLI for a project with multiple models
and tasks.  Trying out tools like click, omegaconf, rich, etc.

One of the problems I've faced when running deep learning experiments on many
_tasks_ (a `LightningDataModule` for example) and models in the same project is
that altough there are many-to-many compatibilities between tasks and models,
_instatiating_ these classes isn't homogeneous.

The compatibility between a task and a model is given by the dataloader's
return type and model's forward method, but many times instantiating models and
tasks differs between the different architectures or data source. Normally I end up with code that looks like this:
```python
if model_name == "lit-rnn":
    model = models.LitRNN(args.vocab_size, args.activation, ...)
elif model_name == "lit-transformer":
    model = models.LitTransformer(args.vocab_size, args.attention_layers, args.projection_size, ...)
elif ...:
    # etc
```
And pretty much the same for the data sources.

Altough a common `__init__` interface could be enforced, I felt that was a hack
around the intrinsic differences between the models instead of robust solution.

In this repo I try to explore a solution to this that leaves me sufficiently
satisfied.


Also, I wanted to allow configurations to come from many sources, namely
hardcoded defaults, config files and command line arguments. Defaults + config
files are useful when developing models and we want to run the same code
repeatedly with the same parameters, while command line options are useful when
we want to quickly try out different configurations. If we want to do a more
thoruogh hyper param search I would write a new command specific for that, but
that is out of scope for this project (for now).

For my use cases there has been a final configuration source which is a special
config requirements for some task model pairs. Sometimes some task model pair
requires specific configurations, and I wanted to be able to write custom code
to "register" this special interaction. I've faced this specifically when
using tokenizers as a param to my data source, where the vocab size of my
model should match the vocab size of the tokenizer, which isn't really a hparam
to my data source, but a _property_ of one of it's components. In my head a
configuration for this would look like
```python
def configure_task_a_model_b(...):
    # ...
    # all the previous configs
    # ...
    tokenizer = tokenizers.Tokenizer.load(task.tokenizer_name)
    # define some arbitrary relation between args
    config.model.vocab_size = tokenizer.vocab_size
    return config
```
Possibly this is a bad design in many other ways, but what's in this repo is
the best solution I could come up with.

And so, the final precedence of config sources is (from least important to
most)
1. hard-coded defaults
2. config files
3. command line options
4. custom task-model configuration

I'm still not to sold on this ordering, maybe the task-model specifics should
come second and allow for overriding in config files and command line. And
maybe I would like to add env vars to the list as well. Nonetheless, for now
this suffices as a proof of concept and this should be easy to change and
expand, since this precedence order is quite explicit in the code [here](link
to cli.py lines).

## install
```console
pip install -e .  # or -e '.[dev]' for dev dependencies included
```

## try it out
### Simple options
Check the available options with
```console
python -m simple_package train --help
```

Check the available task model pairs
```console
python -m simple_package train --task-compatibility
```
Read the help message (using the `--help` flag) for the legend used.

### Running experiments
Use the `-t/--task` and `-m/--model` options to define the task-model pair you
want to run.

First, check the current config used for a specific task-model pair by adding
`--print-config`. This is a convinience flag to check the final config used,
since values can come from different sources.
```console
python -m simple_package train -t lit-complex-args -m lit-conv-net --print-config
```
The `???` value means that said config value hasn't been set, and the
experiment run will fail because the config is incomplete. The values already
there are defaults set in the code.

You can see the error message with the same command as before but without
`--print-config`.
```console
python -m simple_package train -t lit-complex-args -m lit-conv-net
```

To run a succesful experiment you need to provide a complete configuration,
meaning the `???` values should be filled in. To do this there are 2 options
(well 3, but lets not consider _modifying the source code_ a valid option).

First you can simply pass config values using the `-o` option. The syntax used
is the dotted notation from
[omegaconf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dot-list).
To add `determinist=True` to the trainer and `batch_size=32` to the task we
would do
```console
python -m simple_package train -t lit-complex-args -m lit-conv-net -o trainer.deterministic=true -o task.batch_size=32 --print-config
```
See how the values we specified are now peresent in the config.

To simplify the process of configuring the app, we can pass a config file
```console
python -m simple_package train -t lit-complex-args -m lit-conv-net --config-file conf/incomplete-example.yaml --print-config
```
Here the config file defines some missing values, but also overrides some default values (`trainer.max_epochs`) for example. The rest of missing options could be filled in from the command line.

Now lets run a complete example, where the config file contains all the necessary keys but we want to override the batch size from the command line.
```console
python -m simple_package train -t lit-simple-args -m lit-rnn --config-file conf/complete-example.yaml -o task.batch_size=1024
```

The command simply prints how it initialized the task, model and trainer
classes. Since I commonly use Pytorch Lightning, that is all that is needed to
train a model with a datasource, and so it is straightforward how to apply this
CLI structure to an actual `LightningModule` + `LightningDataModule`.


## Epilog

Your are encouraged to read the source code, I left module docstrings in most
so understanding the role of each is easier. Roughly, `cli.py` configures the
CLI and `datasets.py`. `models.py` and `trainer.py` define mock objects of what
would be real objects in a deep learning projects (possibly using Pytorch
Lightning). `training.py` defines training utilities, which for this small
project is only a task-model compatibility matrix, and allows for registering
custom configuration code.

**Disclaimer**: I sprinkled some dynamic classes and metaclasses here and
there, which is totally unnecesary, but I wanted to try out stuff. The same
behaviour could've been accomplished by designing this a bit differentlly.

## TODO
* add module docstrings

