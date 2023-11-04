package main 
 
func AnyInt(x, y, z interface{}) bool {
	if x == y+z || y == x+z || z == x+y {
		return true
	}
	return false
}
