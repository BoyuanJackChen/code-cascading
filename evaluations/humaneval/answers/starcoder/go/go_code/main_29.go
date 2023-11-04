package main 
 
import (          "strings" 
 )

func FilterByPrefix(strings []string, prefix string) []string {
	var filtered []string
	for _, s := range strings {
		if strings.HasPrefix(s, prefix) {
			filtered = append(filtered, s)
		}
	}
	return filtered
}