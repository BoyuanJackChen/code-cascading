package main 
 
func CorrectBracketing(brackets string) bool {
    stack := []string{}
    for _, b := range brackets {
        if b == '(' {
            stack = append(stack, "(")
        } else {
            if len(stack) == 0 {
                return false
            }
            stack = stack[:len(stack)-1]
    }
    return len(stack) == 0
}