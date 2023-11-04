import re

str_with_comments = "# Definition for singly-linked list.\n# class ListNode(object):\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\nclass Solution(object):\n    def addTwoNumbers(self, l1, l2):\n\t\t\"\"\"\n\t\tYou are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.\n\t\tYou may assume the two numbers do not contain any leading zero, except the number 0 itself.\n\t\tExample 1:\n\t\tInput: l1 = [2,4,3], l2 = [5,6,4]\n\t\tOutput: [7,0,8]\n\t\tExplanation: 342 + 465 = 807.\n\t\tExample 2:\n\t\tInput: l1 = [0], l2 = [0]\n\t\tOutput: [0]\n\t\tExample 3:\n\t\tInput: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]\n\t\tOutput: [8,9,9,9,0,0,0,1]\n        :type l1: ListNode\n        :type l2: ListNode\n        :rtype: ListNode\n\t\t\"\"\""

new_str = re.sub(r'^\s*#.*\n', '', str_with_comments, flags=re.MULTILINE)

print(new_str)
# 问题出在744