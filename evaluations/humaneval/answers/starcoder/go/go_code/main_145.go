package main 
 
import (          "strconv" 
          "sort" 
 )

func OrderByPoints(nums []int) []int {
    type pair struct {
        num int
        points int
    }
    var pairs []pair
    for i, num := range nums {
        points := 0
        for _, digit := range strconv.Itoa(num) {
            points += int(digit) - '0'
        }
        pairs = append(pairs, pair{num, points})
    }
    sort.Slice(pairs, func(i, j int) bool {
        if pairs[i].points == pairs[j].points {
            return pairs[i].num < pairs[j].num
        }
        return pairs[i].points < pairs[j].points
    })
    var result []int
    for _, pair := range pairs {
        result = append(result, pair.num)
    }
    return result
}
