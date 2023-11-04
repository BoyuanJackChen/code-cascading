package main 
 
import (          "strings" 
 )

func SameChars(s0 string, s1 string) bool {
	return strings.Count(s0, s1[0]) == strings.Count(s1, s1[0])
}
