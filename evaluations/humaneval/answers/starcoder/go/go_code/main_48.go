package main 
 
import (          "strings" 
 )

func IsPalindrome(text string) bool {
	return strings.ToLower(text) == strings.ToLower(Reverse(text))
}