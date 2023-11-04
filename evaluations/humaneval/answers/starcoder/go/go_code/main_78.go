package main 
 
func HexKey(num string) int {
	var count int
	for _, v := range num {
		if isPrime(int(v)) {
			count++
		}
	}
	return count
}