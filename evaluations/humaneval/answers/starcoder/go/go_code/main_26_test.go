package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestRemoveDuplicates(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{}, RemoveDuplicates([]int{}))
    assert.Equal([]int{1, 2, 3, 4}, RemoveDuplicates([]int{1, 2, 3,4}))
    assert.Equal([]int{1, 4, 5}, RemoveDuplicates([]int{1, 2, 3, 2,4, 3, 5}))
}
