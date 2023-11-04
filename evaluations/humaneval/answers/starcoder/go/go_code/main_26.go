package main 
 
func RemoveDuplicates(numbers []int) []int {
	var result []int
	for _, number := range numbers {
		if!contains(result, number) {
			result = append(result, number)
		}
	}
	return result
}