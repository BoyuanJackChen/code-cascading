package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestLargestSmallestIntegers(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([2]interface{}{nil, 1}, LargestSmallestIntegers([]int{2, 4, 1, 3, 5, 7}))
    assert.Equal([2]interface{}{nil, 1}, LargestSmallestIntegers([]int{2, 4, 1, 3, 5, 7, 0}))
    assert.Equal([2]interface{}{-2, 1}, LargestSmallestIntegers([]int{1, 3, 2, 4, 5, 6, -2}))
    assert.Equal([2]interface{}{-7, 2}, LargestSmallestIntegers([]int{4, 5, 3, 6, 2, 7, -7}))
    assert.Equal([2]interface{}{-9, 2}, LargestSmallestIntegers([]int{7, 3, 8, 4, 9, 2, 5, -9}))
    assert.Equal([2]interface{}{nil, nil}, LargestSmallestIntegers([]int{}))
    assert.Equal([2]interface{}{nil, nil}, LargestSmallestIntegers([]int{0}))
    assert.Equal([2]interface{}{-1, nil}, LargestSmallestIntegers([]int{-1, -3, -5, -6}))
    assert.Equal([2]interface{}{-1, nil}, LargestSmallestIntegers([]int{-1, -3, -5, -6, 0}))
    assert.Equal([2]interface{}{-3, 1}, LargestSmallestIntegers([]int{-6, -4, -4, -3, 1}))
    assert.Equal([2]interface{}{-3, 1}, LargestSmallestIntegers([]int{-6, -4, -4, -3, -100, 1}))
}
