package main 
 
func SumSquares(lst []int) int {
    for i := 0; i < len(lst); i++ {
        if i % 3 == 0 {
            lst[i] = lst[i] * lst[i]
        } else if i % 4 == 0 {
            lst[i] = lst[i] * lst[i] * lst[i]
        }
    }
    return sum(lst)
}