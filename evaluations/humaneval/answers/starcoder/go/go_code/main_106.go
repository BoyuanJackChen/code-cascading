package main 
 
func F(n int) []int {
	var result []int
	for i := 1; i <= n; i++ {
		if i%2 == 0 {
			result = append(result, factorial(i))
		} else {
			result = append(result, sum(i))
		}
	}
	return result
}