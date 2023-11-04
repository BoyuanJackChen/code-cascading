package main 
 
func MakePalindrome(str string) string {
    if IsPalindrome(str) {
        return str
    }
    for i := len(str); i >= 0; i-- {
        if IsPalindrome(str[:i]) {
            return str[:i] + Reverse(str[i:])
        }
    }
    return ""
}
func Reverse(str string) string {
    runes := []rune(str)
    result := make([]rune, 0)
    for i := len(runes) - 1; i >= 0; i-- {
        result = append(result, runes[i])
    }
    return string(result)
}