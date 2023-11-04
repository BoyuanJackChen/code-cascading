theirs_34B = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 78, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 112, 114, 116, 117, 122, 123, 126, 128, 133, 136, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162]
theirs_13B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 79, 82, 85, 86, 87, 89, 92, 94, 95, 96, 97, 98, 103, 105, 107, 112, 114, 116, 121, 123, 124, 125, 128, 133, 136, 141, 142, 143, 144, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 161, 162]
my_13B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 76, 79, 82, 85, 86, 87, 89, 92, 94, 95, 96, 97, 98, 103, 105, 107, 112, 114, 116, 121, 123, 124, 125, 128, 133, 136, 141, 142, 143, 144, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 161, 162]
my_34B = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 69, 70, 71, 72, 73, 74, 76, 78, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106, 107, 112, 114, 116, 117, 122, 123, 126, 128, 133, 136, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162]

print(len(theirs_34B))
print(len(my_34B))
my_wrong = []
for num in theirs_34B:
    if num not in my_34B:
        my_wrong.append(num)
print(f"34B my wrong their correct: {my_wrong}")
their_wrong = []
for num in my_34B:
    if num not in theirs_34B:
        their_wrong.append(num)
print(f"34B their wrong my correct: {their_wrong}")

print(len(theirs_13B))
print(len(my_13B))
my_wrong = []
for num in theirs_13B:
    if num not in my_13B:
        my_wrong.append(num)
print(f"13B my wrong their correct: {my_wrong}")
their_wrong = []
for num in my_13B:
    if num not in theirs_13B:
        their_wrong.append(num)
print(f"13B their wrong my correct: {their_wrong}")


# 10, mine
#Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
#Create a Python script for this problem:


def is_palindrome(string: str) -> bool:
        """ Test if given string is a palindrome """
        return string == string[::-1]


def make_palindrome(string: str) -> str:
        """ Find the shortest palindrome that begins with a supplied string.
        Algorithm idea is simple:
        - Find the longest postfix of supplied string that is a palindrome.
        - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
        >>> make_palindrome('')
        ''
        >>> make_palindrome('cat')
        'catac'
        >>> make_palindrome('cata')
        'catac'
        """


### Response:
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string. """
    if is_palindrome(string):
        return string
    
    for i in range(len(string)-1, -1, -1):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
    
    return string + string[::-1]

# Theirs
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string. """
    if is_palindrome(string):
        return string
    
    for i in range(len(string)-1, -1, -1):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
    
    return string + string[::-1]
