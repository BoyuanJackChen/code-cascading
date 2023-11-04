package main 
 
import (          "fmt" 
 )

func main() {
	var lst []int
	var sum int
	lst = []int{5, 8, 7, 1}
	sum = 0
	for i := 0; i < len(lst); i++ {
		if i%2 == 0 {
			sum += lst[i]
		}
	}
	fmt.Println(sum)
}
