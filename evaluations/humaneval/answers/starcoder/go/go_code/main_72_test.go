package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestWillItFly(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, WillItFly([]int{3, 2, 3}, 9))
    assert.Equal(false, WillItFly([]int{1, 2}, 5))
    assert.Equal(true, WillItFly([]int{3}, 5))
    assert.Equal(false, WillItFly([]int{3, 2, 3}, 1))
    assert.Equal(false, WillItFly([]int{1, 2, 3}, 6))
    assert.Equal(true, WillItFly([]int{5}, 5))
}
