package main 
 
func GenerateIntegers(a, b int) []int {
	var result []int
	for i := a; i <= b; i++ {
		if i%2 == 0 {
			result = append(result, i)
		}
	}
	return result
}