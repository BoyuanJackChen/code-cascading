from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import accelerate
import torch
from model_classifier import MLPClassifier, LSTMClassifier

accelerator = accelerate.Accelerator()

# device = torch.device('cuda:7')
stop_words = ["\n\n", "\n \n", "\n  \n", "\n   \n", "\n    \n"]
#0
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
#31
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
#35
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
#140
prompt_140 = "\ndef fix_spaces(text):\n    \"\"\"\n    Given a string text, replace all spaces in it with underscores, \n    and if a string has more than 2 consecutive spaces, \n    then replace all consecutive spaces with - \n    \n    fix_spaces(\"Example\") == \"Example\"\n    fix_spaces(\"Example 1\") == \"Example_1\"\n    fix_spaces(\" Example 2\") == \"_Example_2\"\n    fix_spaces(\" Example   3\") == \"_Example-3\"\n    \"\"\"\n"
#141
prompt_141 = "\ndef file_name_check(file_name):\n    \"\"\"Create a function which takes a string representing a file's name, and returns\n    'Yes' if the the file's name is valid, and returns 'No' otherwise.\n    A file's name is considered to be valid if and only if all the following conditions \n    are met:\n    - There should not be more than three digits ('0'-'9') in the file's name.\n    - The file's name contains exactly one dot '.'\n    - The substring before the dot should not be empty, and it starts with a letter from \n    the latin alphapet ('a'-'z' and 'A'-'Z').\n    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']\n    Examples:\n    file_name_check(\"example.txt\") # => 'Yes'\n    file_name_check(\"1example.dll\") # => 'No' (the name should start with a latin alphapet letter)\n    \"\"\"\n"
#148
prompt_148 = "\ndef bf(planet1, planet2):\n    '''\n    There are eight planets in our solar system: the closerst to the Sun \n    is Mercury, the next one is Venus, then Earth, Mars, Jupiter, Saturn, \n    Uranus, Neptune.\n    Write a function that takes two planet names as strings planet1 and planet2. \n    The function should return a tuple containing all planets whose orbits are \n    located between the orbit of planet1 and the orbit of planet2, sorted by \n    the proximity to the sun. \n    The function should return an empty tuple if planet1 or planet2\n    are not correct planet names. \n    Examples\n    bf(\"Jupiter\", \"Neptune\") ==> (\"Saturn\", \"Uranus\")\n    bf(\"Earth\", \"Mercury\") ==> (\"Venus\")\n    bf(\"Mercury\", \"Uranus\") ==> (\"Venus\", \"Earth\", \"Mars\", \"Jupiter\", \"Saturn\")\n    '''\n"
#152
prompt_152 = "\ndef compare(game,guess):\n    \"\"\"I think we all remember that feeling when the result of some long-awaited\n    event is finally known. The feelings and thoughts you have at that moment are\n    definitely worth noting down and comparing.\n    Your task is to determine if a person correctly guessed the results of a number of matches.\n    You are given two arrays of scores and guesses of equal length, where each index shows a match. \n    Return an array of the same length denoting how far off each guess was. If they have guessed correctly,\n    the value is 0, and if not, the value is the absolute difference between the guess and the score.\n    \n    \n    example:\n\n    compare([1,2,3,4,5,1],[1,2,3,4,2,-2]) -> [0,0,0,0,3,3]\n    compare([0,5,0,0,0,4],[4,1,1,0,0,-2]) -> [4,4,1,0,0,6]\n    \"\"\"\n    return [0 if guess[i] == game[i] else abs(guess[i] - game[i]) for i in range(len(game))]\n\nprint(compare([1,2,3,4,5,1],[1,2,3,4,2,-2]))\nprint(compare([0,5,0,0,0,4],[4,1,1,0,0,-2]))\n<|endoftext|>"
#161
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def trim_with_stopwords(outputs, stopwords, original_prompt) -> str:
    result = []
    len_prompt = len(original_prompt)
    for output in outputs:
        answer = output[len_prompt:]
        min_i = len(answer)
        for w in sorted(stopwords, reverse=True):
            for i in range(len(answer)):
                if answer[i:].startswith(w) and min_i > i:
                    min_i = i
        answer = answer[:min_i]
        result.append(answer)
    return result


if __name__== "__main__":
    checkpoint = "Salesforce/codegen-16B-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

    ee_model = LSTMClassifier()
    ee_model = torch.load("../ee_classifier/checkpoints_saved/w5/epoch_18.pt")
    ee_model = accelerator.prepare(ee_model)
    ee_model.eval()
    print("ee model loaded")

    model.set_want_exited(True)
    model.set_want_hidden_states(True)
    model.set_ee_sequential(ee_model)
    model.set_ee_threshold(0.98)

    stop_words = ["\n\n", "def", "print"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

    for prompt in [prompt_140, prompt_148, prompt_152, prompt_161]:
        # Remove all the consecutive newlines at the beginning
        if prompt.startswith("\n"):
            prompt = prompt.lstrip("\n")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        # Stopping Criteria
        encounters = 1
        for stop_word_id in stop_words_ids:
            stop_count = (stop_word_id == input_ids[0]).sum().item()
            encounters += stop_count
        # print(f"input_ids is {input_ids}")
        # print(f"stop_words_ids is {stop_words_ids}")
        # print(f"Final encounters is {encounters}")
        print(f"The final encounter is {encounters}")
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters)])

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature = 0.1,
            top_k = 0,
            top_p = 0.9,
            # num_beams = 4,
            # num_beam_groups = 4,
            # diversity_penalty=0.3,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            stopping_criteria=stopping_criteria
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_list = []
        for ids in generated_ids[0]:
            word = tokenizer.decode(int(ids))
            decoded_list.append(word)

        # Get the start and end index in decoded list
        start_index = 0
        end_index = len(decoded_list)
        print(generated_text[0], "\n")

        # Print decoded list in the range
        print(f"decoded list is \n{decoded_list}")

        # Get early exit layer indices
        all_layer_indices = model.get_early_exit_layer_indices().tolist()
        # Throw away all -1 elements in all_layer_indices
        actual_layer_indices = [x for x in all_layer_indices if x != -1]
        # print(f"start_index is {start_index}; end_index is {end_index}")
        print(f"Actual early exit layer indices are \n{actual_layer_indices}; \n"
            f"average is {sum(actual_layer_indices)/len(actual_layer_indices)}")

        model.clear_early_exit_layer_indices()


# w5/epoch18: 0.99 saves 4.0 layers with full accuracy
# w10/epoch27: 0.995 saves 4.2 layers with full accuracy
# w3/epoch5: 0.9 saves 3.6 layers with good accuracy