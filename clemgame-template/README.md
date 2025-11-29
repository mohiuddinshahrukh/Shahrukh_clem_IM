# Template for creating new `clemgames'

This repository provides example code for starting to develop new games for the `clemcore` environment, as used in the `clembench` project.

[documentation to be further adapted]

- clemcore (can be installed via `pip`, access to this code is normally not needed for game development): <https://github.com/clembench/clemcore>
- clembench (more examples of games, not all of which however are making use of the newest features, which is why it is better to start with the code in the present repository): <https://github.com/clembench/clembench>


### Evaluation Results

On the [main project website](https://clembench.github.io) , under [leaderboard](https://clembench.github.io/leaderboard.html).

# How to Prototype Games

We have linked here the documentation on how to prototype games, needed basic knowledge and sample code under these notebooks:

[How to make a clemgame](https://github.com/clp-research/clemcore/blob/main/docs/howto_make_a_clemgame.ipynb)

[How to prototype games](https://github.com/clp-research/clemcore/blob/main/docs/howto_prototype_games.ipynb)




# Using the clemcore CLI

Create a separate virtual environment in Python (3.10, 3.11, 3.12) and activate it.

```
(myclem) python3 -m venv venv

(myclem) source venv/bin/activate
```

Then install `clemcore`:
```
(myclem) pip install clemcore
``` 

To use local GPUs (via `transformers` or `vllm` libraries etc.) install the following libraries:
```
(myclem) pip install clemcore[huggingface] # dependencies for the local hf backend
(myclem) pip install clemcore[vllm]        # dependencies for the local vllm backend
(myclem) pip install clemcore[slurk]       # dependencies for the slurk backend 
```

After the installation you will have access to the `clem` CLI tool. The main functions are:

```
(myclem) clem list games               # list the games available for a run
(myclem) clem list backends            # list the backends available for a run
(myclem) clem list models              # list the models available for a run
(myclem) clem run -g <game> -m <model> # runs the game benchmark; also transcribes and scores
(myclem) clem transcribe               # translates interactions into html files
(myclem) clem score                    # computes individual performance measures
(myclem) clem eval                     # computes overall performances measures; requires scores
```

Note that `clem` operates relative to the current working directory, that is, the directory it is called from.
This directory is what we call the workspace.
A workspace may look like this.

```
(optional) key.json
(optional) game_registry.json 
(optional) model_registry.json  
(optional) custom_api.py 
clembench/
```

The files have the following functions:
- **key.json**: contains the secrets for the remote api calls; if this file does not exist, then `clem` looks into `~/.clemcore/`
- **game_registry.json**: allows to make additional game specifications useable for the runs. The game specifications must at least contain the `game_name`, `game_path` and `players` attribute. We provide the template file.
- **model_registry.json**: allows to add additional model specifications. This is specifically useful to run with models that have not been packaged yet. In addition, it allows to point model specification to custom backend names.
- **custom_api.py**: `clem` automatically discovers additional _api files placed into the cwd, so that users of the framework can run their own backends with the games.
- **clembench/**: contains the game directories (with the game code) available for the benchmark runs

Note that, `clem` does now automatically discovers game directories that are at most 3-levels away from the `cwd`. 
To be discoverable, directories have to carry a `clemgame.json` (here a game path is not required, because `clem` automatically determines it).

## Use Case: Game Developer

As a game developer you want to implement your own game to be run with `clem`.
You will use a typical clem game project structure.
The game directory will become your workspace.
To make the game visible to `clem` you need to add a `clemgame.json` to the directory.
This file should specify at least the following
```
{
"game_name": "mygame",
"description": "A brief description of mygame",
"player": "single" | "two" | "multi",
"image": "none" | "single" | "multi",
"languages": ["en"]
}
```

To test your game with some packaged models, you will add a `key.json` and run the command `clem run -g mygame -m model` from within the game directory.
The results will be written into `results`.
To also get html transcripts you can run `clem transcribe -g mygame`.
Overall, a game developers workspace directory will possibly look as follows:

```
mygame
- in/
- resources/
- results/
- __init__.py
- master.py
- instancegenerator.py
- clemgame.json   
```

## Use Case: Running Games

We include an example game `taboo` under `clembench` directory (the game includes `clemgame.json` file and `clem` can detect it as a game.

### List games
Let's check the available games in the current directory:


```
(myclem) clem list games
```
We get this output:

<pre>
Found '1' game specs that match the game_selector='all'

taboo:
	Taboo game between two agents where one has to describe a word for the other to guess.
</pre>

### List models
Let's check which models (LLMs) are supported:

```
(myclem) clem list models
```

We get this output:

<pre>
Found '180' registered model specs:
...
gpt-4o-2024-08-06 -> openai (packaged)
gpt-4o-mini-2024-07-18 -> openai (packaged)
...
o1-mini-2024-09-12 -> openai (packaged)
o3-mini-2025-01-31 -> openai (packaged)
...
claude-3-5-haiku-20241022 -> anthropic (packaged)
claude-3-5-sonnet-20241022 -> anthropic (packaged)
claude-3-7-sonnet-20250219 -> anthropic (packaged)
...
gemini-2.0-flash-exp -> google (packaged)
...
Meta-Llama-3.1-8B-Instruct -> huggingface_local (packaged)
Meta-Llama-3.1-70B-Instruct -> huggingface_local (packaged)
...
InternVL2-40B -> huggingface_multimodal (packaged)
InternVL2-26B -> huggingface_multimodal (packaged)
...
</pre>

### Running and benchmarking

Let's run the existing `taboo` game using `gpt-4o-2024-08-06`. To run such commercial model from OpenAI, it is required to have `key.json` file with the API and organization keys filled:

```
(myclem) clem run -g taboo -m gpt-4o-2024-08-06       # runs all experiments
```

The results will be written into `results` folder. It also creates a log file (clembench.log) for all episodes that were run. 


Then run the following script that generates the transcript for each game play (episode), it generates `transcript.html` and `transcript.tex` files under each episode directory:

```
(myclem) clem transcribe               # translates interactions into html files for all games that are accessible.
```

Finally, run `benchmark eval`:

```
(myclem) clem eval                     # computes overall performances measures
```

It generates `results.html`, `results.csv` and `raw.csv` files inside the `results` directory. `results.html` or `results.csv` compares all models that were run. `raw.csv` is the dump of all averaged scores (episode and turn) that can be used for further analysis.

**Benchmarking script**: We also provided a script to benchmark any implemented game with multiple models. See `run_benchmark.sh` and adjust the script where needed.



#### More options

To see more options for running a game, execute the following script:

```
(myclem) clem run -h       # help
```

We get the following output:
<pre>
options:
  -h, --help            show this help message and exit
  -m [MODELS ...], --models [MODELS ...]
                        Assumes model names supported by the implemented backends.
                        
                              To run a specific game with a single player:
                              $> clem run run -g privateshared -m mock
                        
                              To run a specific game with a two players:
                              $> clem run -g taboo -m mock mock
                        
                              If the game supports model expansion (using the single specified model for all players):
                              $> clem run -g taboo -m mock
                        
                              When this option is not given, then the dialogue partners configured in the experiment are used. 
                              Default: None.

  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        Optional argument to only run a specific experiment

  -g GAME, --game GAME  A specific game name (see ls), or a GameSpec-like JSON string object.

  -t TEMPERATURE, --temperature TEMPERATURE
                        Argument to specify sampling temperature for the models. Default: 0.0.

  -l MAX_TOKENS, --max_tokens MAX_TOKENS
                        Specify the maximum number of tokens to be generated per turn (except for cohere). Be careful with high values which might lead to exceed your API token limits. Default: 100.

  -i INSTANCES_NAME, --instances_name INSTANCES_NAME
                        The instances file name (.json suffix will be added automatically.

  -r RESULTS_DIR, --results_dir RESULTS_DIR
                        A relative or absolute path to the results root directory. For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. When not specified, then the results will be located in 'results

</pre>

For instance, we can run one specific experiment `high_en` in `taboo` like this:

 ```
(myclem) clem run -g taboo -e high_en -m gpt-4o-2024-08-06       # runs only "high_en" experiment
```

Since `taboo` is a two player game (one player gives clues and the other player guesses the word), we can make two different models play this game, the first one is a clue giver (gpt-4o-2024-08-06) and the second is a guesser (gpt-4o-mini-2024-07-18):


 ```
(myclem) clem run -g taboo -m gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18
```

Or simply run the mock version of models (produces some static output) just to test the setup.

```
(myclem) clem run -g taboo -m mock
```


##### Multiple instance files

By default, the `clem run -g game -m model` runs the `instances.json` file under `game_dir/in/instances.json`. You can run different instance files with the `-i instances_of_your_choice` (.json is added automatically). 

Let's say you want to benchmark certain models on your multilingual abilities and your game supports those languages (German, English, etc.). Then, you can include a separate instance file for each language and encapsulates all needed information for that language under `game_dir/in` directory as such: `instances_de.json` and `instances_en.json`


```
(myclem) clem run -g GAME_NAME -m MODEL_NAME -i instances_de -r results_de       # runs specific instance file and saves the results under specific folder
```
and

```
(myclem) clem run -g GAME_NAME -m MODEL_NAME -i instances_en -r results_en
```
Doing it this way, the results are saved in separate folders (if that is what you wish, otherwise remove -r option).

Running `eval` script by default looks for `results` folder. If you specified different results folder, simply run with the -r option:

```
(myclem) clem transcribe -r results_en
(myclem) clem transcribe -r results_de

(myclem) clem eval -r results_en
(myclem) clem eval -r results_de
```



## Running Local Models

### GPU
The framework already supports multiple models that are hosted on HuggingFace (check with `clem list models`).

To run any supported model (e.g. `Meta-Llama-3.1-8B-Instruct`), simply run the following script:

```
(myclem) clem run -g taboo -m Meta-Llama-3.1-8B-Instruct
```

It will start downloading the model weights from HF, once downloaded the game will be played. GPUs are required to run such models.


### LlamaCPP

### VLLM

### OpenAI Compatible Server

If you have a local model hosted on your computer (Ollama, etc.) and it is possible to query it via API, then simply add your API details (base URL and API key if required) to the `key.json` file under `generic_openai_compatible` field:

Let's say the model is accessbile with these parameters:

```
"generic_openai_compatible": {"api_key":"0123456", "base_url": "http://localhost:8080/query"}
```

Then add the model details to the `model_registry.json` file. We provided an example (my_model) in the template file (model_registry.json.template).

To run the model, as usual:

```
(myclem) clem run -g taboo -m my_model
```

Similarly, if you have access to other OpenAI compatible models, simply add the base URL and your key as above and you can send the queries to that server/model.

