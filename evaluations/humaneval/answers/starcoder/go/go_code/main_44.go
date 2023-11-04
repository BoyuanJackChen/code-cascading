package main 
 
func ChangeBase(x int, base int) string {
    var result string
    var remainder int
    var power int
    for x > 0 {
        remainder = x % base
        x = x / base
        result = string(remainder+48) + result
        power++
    }
    return result
}