import multiprocessing

answer_textcode = """
def is_Fibbonacci(n):
    if n == 0:
        return False
    elif n == 1:
        return True
    else:
        return is_Fibbonacci(n-1) + is_Fibbonacci(n-2)
assert is_Fibbonacci(2) == True
assert is_Fibbonacci(319) == False
# assert is_Fibbonacci(2971215073) == True
# assert is_Fibbonacci(2971215074) == False
# assert is_Fibbonacci(6684390626) == True
# assert is_Fibbonacci(6684390627) == True
    """

correct = False
def code_to_run(result_queue):
    try:
        exec(answer_textcode, globals())
        print("No error occurred!\n")
        result_queue.put(True)
    except Exception as e:
        print(f"Error occurred: {e}\n")
        result_queue.put(False)

result_queue = multiprocessing.Queue()
process = multiprocessing.Process(target=code_to_run, args=(result_queue,))
process.start()
process.join(3)  # wait up to 3 seconds

if process.is_alive():
    print("Code took too long to run!")
    process.terminate()
    correct = False
else:
    correct = result_queue.get()

if correct:
    print("Correct!")
else:
    print("Incorrect!")
